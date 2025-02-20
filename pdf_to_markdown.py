import os
from pathlib import Path
import base64
from openai import AzureOpenAI
from pdf2image import convert_from_bytes
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from datetime import datetime
import io
import math
import time
import psutil
from PIL import Image, ImageEnhance
from PyPDF2 import PdfReader, PdfWriter
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat

class ConverterByGPT:
    def __init__(self, job_id: str):
        # Initialize Azure OpenAI with key-based authentication
        self.client = AzureOpenAI(
            azure_endpoint=OCR_AZURE_OPENAI_ENDPOINT,
            api_key=OCR_AZURE_OPENAI_KEY,
            api_version=OCR_AZURE_OPENAI_API_VERSION,
        )

        self.temp_dir = f"{TEMP_DIR}/{job_id}"
        
        # Create necessary directories
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        self.current_date = datetime.now().strftime("%m/%d/%Y")

    def split_pdf_to_images(self, pdf_content: bytes):
        """Convert PDF pages to PNG images"""
        images = convert_from_bytes(pdf_content, grayscale=True)
        image_paths = []
        
        for i, image in enumerate(images):
            image_path = f"{self.temp_dir}/page_{i+1}.png"
            
            # Convert to grayscale and enhance contrast
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(2.0)  # Increase contrast by a factor of 2.0
            
            # Compress image before saving
            compressed_image = self.compress_image(enhanced_image)
            compressed_image.save(image_path, "PNG")
            
            # Verify file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            print(f"Page {i+1} size: {file_size_mb:.2f} MB")
            
            if file_size_mb > MAX_IMAGE_SIZE_MB:
                print(f"Warning: Page {i+1} is still over {MAX_IMAGE_SIZE_MB}, applying emergency compression")
                with Image.open(image_path) as img:
                    extra_compressed = self.compress_image(img, target_size_mb=TARGET_IMAGE_SIZE_MB)  # Target slightly below {TARGET_IMAGE_SIZE_MB}
                    extra_compressed.save(image_path, "PNG")
                    final_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                    print(f"Final size after emergency compression: {final_size_mb:.2f} MB")
            
            image_paths.append(image_path)
        
        return image_paths

    def compress_image(self, image, target_size_mb=MAX_IMAGE_SIZE_MB):
        """Compress image to target size of {MAX_IMAGE_SIZE_MB} with verification"""
        def get_size_mb(img):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            return len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        current_size_mb = get_size_mb(image)
        if current_size_mb <= target_size_mb:
            return image
        
        # First attempt: JPEG compression
        quality = 95
        compressed_image = image
        while current_size_mb > target_size_mb and quality > 5:
            img_byte_arr = io.BytesIO()
            if image.mode in ('RGBA', 'P'):
                compressed_image = image.convert('RGB')
            else:
                compressed_image = image
            
            compressed_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            current_size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
            quality -= 10  # More aggressive quality reduction
        
        # Second attempt: Resize if still too large
        scale_factor = 1.0
        while current_size_mb > target_size_mb and scale_factor > 0.1:
            scale_factor *= 0.7  # More aggressive scaling
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            compressed_image = compressed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_size_mb = get_size_mb(compressed_image)
        
        # Final verification
        final_size = get_size_mb(compressed_image)
        print(f"Final compressed image size: {final_size:.2f} MB")
        if final_size > target_size_mb:
            # Emergency compression: force resize to ensure size limit
            scale_factor = math.sqrt(target_size_mb / final_size) * 0.9  # 10% safety margin
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            compressed_image = compressed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Log final compressed size
        final_size = get_size_mb(compressed_image)
        print(f"Emergency compression result size: {final_size:.2f} MB")
        return compressed_image
    
    def retry_with_backoff(self, func, max_retries = RATE_LIMIT_RETRY_MAX_COUNT, base_delay = RATE_LIMIT_RETRY_DELAY):
        """Retries a function with exponential backoff in case of 429 errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if "429" in str(e):
                    wait_time = base_delay * (2 ** attempt)
                    print(f">>>> Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(">>>> Max retries exceeded due to rate limiting.")

    def image_to_markdown(self, image_info):
        """Convert image to markdown using Azure OpenAI"""
        image_path, page_num = image_info
        print(f"Processing page {page_num + 1}...")
        
        # Enhanced system prompt focusing on accuracy
        system_prompt = f"""
The current date is: {self.current_date}. 
You are an expert document and form extractor at a law firm. 
Your job is to meticulously extract all the information from this image.

For every field you extract:
1. Note if this is filled out in handwritten form or typed
2. For checkboxes:
   - Mark [X] ONLY when you are certain the box is checked (contains clear marks, X, or checkmark)
   - Mark [ ] when the box is clearly empty
   - If you're unsure about a checkbox status, note "(Status unclear)"

3. For text and numbers:
   - Extract exactly as they appear, preserving original formatting
   - If text is unclear, mark as "(Unclear: possible text)"
   - For empty fields, mark as "[Field is blank]"
   - For unreadable fields, mark as "(Unreadable)"

4. Form Structure:
   - Maintain exact form section headers and numbering
   - Include all field labels exactly as they appear
   - Preserve the hierarchy of sections

DO NOT:
- Guess or infer information that isn't clearly visible
- Mark checkboxes as checked unless you're absolutely certain
- Modify or "correct" any information - extract exactly as shown

Follow the sample output format provided below.

**Document Name:** Cover Letter

**Extracted Information:**

- **Date:** August 15, 2018 (Typed)
- **Sender Information:**
- Name: Maureen Macroscopus (Typed)
- Address: 1234 Main Street, Los Angeles, CA 90002 (Typed)
- Phone: (213) 555-1232 (Typed)
- Email: immigrationlady@abcus.com (Typed)

- **Recipient Information:**
- USCIS (Typed)
- P.O. Box 21262 (Typed)
- Phoenix, AZ 85036 (Typed)

- **Subject:** Form I-90, Application to Replace Permanent Resident Card (Renewal) (Typed)
- **Applicant:** Serena Janice Williams (Typed)
- **Alien No:** 210-123-456 (Typed)

- **Filing Fee:**
- A. Cashier's Check in the amount of $540.00 payable to the U.S. Department of Homeland Security (Typed)

- **Photographs:**
- B. Two Passport Style Photographs of Applicant (Typed)

- **Forms:**
- C. I-90, Application to Replace Permanent Resident Card (Typed)

- **Identity and Proof of Citizenship/Nationality Documents:**
- D. Copy of Applicant's American Passport (Typed)
- E. Copy of Front and Back of Applicant's LPR Card (Typed)

- **Signature:** Signature present (Handwritten)

**Instructions:**
- The letter is a cover letter for the application packet for replacing a Permanent Resident Card.
- It lists the documents and items included in the application packet.
- It instructs not to contact the applicant with any questions or concerns.
---
**Document Name:** Exhibit A

This page is a cover page titled "EXHIBIT - A". There is no additional information, fields, or instructions to extract from this page.
---
**Document Name:** Filing Fees Instructions

Instructions Extracted:
1. Filing Fees
2. Remember to:
- Use a Cashier's Check, Money Order, or Personal Check.
- Make it payable to: US Department of Homeland Security.
- Include Name and A# in the Memo Line.

No fields to fill out on this page.
---
Document Name: Exhibit B

Information Extracted:
- Title: EXHIBIT - B

Note: This page appears to be a cover page or a divider for Exhibit B. No fields to fill out or instructions are present on this page.
---
Document Name: Passport Style Photo Instructions

Instructions Extracted:
1. Label Each Photo with full name and A#.
2. Photos should be 2 x 2.
3. Photos MUST have white background.
4. Label the Envelope.

No fields to fill out on this page.
---
Document Name: Exhibit C

Information Extracted:
- The page is a cover page for Exhibit C.
- No fields to fill out or instructions are present on this page.
- No signature field is present.
---
**Document Name:** Application to Replace Permanent Resident Card (Form I-90)
**Page Number:** 1 of 7

### Extracted Information:

#### Part 1. Information About You
1. **Alien Registration Number (A-Number):** A 0 9 8 7 6 5 4 3 2 (Handwritten)
2. **USCIS Online Account Number (if any):** [Field is blank]
3. **Your Full Name:**
- **Family Name (Last Name):** WILLIAMS (Typed)
- **Given Name (First Name):** ANDREA (Typed)
- **Middle Name:** JAMES (Typed)

4. **Has your name legally changed since the issuance of your Permanent Resident Card?**
- [ ] Yes (Proceed to Item Numbers 5.a. - 5.c.)
- [x] No (Proceed to Item Numbers 6.a. - 6.c.) (Typed)
- [ ] Yes, I am a commuter and my name has legally changed since the issuance of my Permanent Resident Card. (Proceed to Item Numbers 5.a. - 5.c.)

5. **Provide your name exactly as it is printed on your current Permanent Resident Card:**
- **Family Name (Last Name):** [Field is blank]
- **Given Name (First Name):** [Field is blank]
- **Middle Name:** [Field is blank]

#### Mailing Address
6.a. **In Care Of Name:** [Field is blank]
6.b. **Street Number and Name:** 1234 PALMER STREET (Typed)
- [x] Apt. [ ] Ste. [ ] Flr.
6.c. **City or Town:** ENGLEWOOD (Typed)
6.d. **State:** CA (Typed)
6.e. **ZIP Code:** 90210 (Typed)
6.f. **Province:** [Field is blank]
6.g. **Postal Code:** [Field is blank]
6.h. **Country:** USA (Typed)

#### Physical Address
(Provide this information only if different from mailing address.)
7.a. **Street Number and Name:** SAME AS ABOVE (Typed)
- [ ] Apt. [ ] Ste. [ ] Flr.
7.b. **City or Town:** [Field is blank]
7.c. **State:** [Field is blank]
7.d. **ZIP Code:** [Field is blank]
7.e. **Province:** [Field is blank]
7.f. **Postal Code:** [Field is blank]
7.g. **Country:** [Field is blank]

### Instructions:
- **Type or print in black ink.**
- **If your name has legally changed since the issuance of your Permanent Resident Card, proceed to Item Numbers 5.a. - 5.c.**
- **If your name has not legally changed, proceed to Item Numbers 6.a. - 6.c.**
- **Provide your name exactly as it is printed on your current Permanent Resident Card.**
- **Provide physical address only if different from mailing address.**

### Signature:
- No signature field is present on this page.
---
**Document Name:** Form I-90
**Page Number:** 2 of 7

### Extracted Information:

#### Part 1. Information About You (continued)

- **Gender:** Female (Typed)
- **Date of Birth (mm/dd/yyyy):** 04/15/2000 (Typed)
- **City/Town/Village of Birth:** HARTFORD (Typed)
- **Country of Birth:** AMERICA (Typed)

- **Mother's Name:**
- **Given Name (First Name):** JANE (Typed)
- **Family Name (Last Name):** DOE (Typed)

- **Father's Name:**
- **Given Name (First Name):** JOHN (Typed)
- **Family Name (Last Name):** DOE (Typed)

- **Class of Admission:** (Not filled)
- **Date of Admission (mm/dd/yyyy):** 06/15/2005 (Typed)
- **U.S. Social Security Number (if any):** 123-45-6789 (Typed)

#### Part 2. Application Type

- **My status is (Select only one box):**
- [ ] Lawful Permanent Resident (Proceed to Section A.)
- [X] Permanent Resident - In Commuter Status (Proceed to Section B.) (Typed)
- [ ] Conditional Permanent Resident (Proceed to Section C.)

### Instructions:

- **Reasons for Application (Select only one box):**
- Section A: (To be used only by a lawful permanent resident or a permanent resident in commuter status.)
- [ ] My previous card has been lost, stolen, or destroyed.
- [ ] My previous card was issued but never received.
- [ ] My existing card has been mutilated.
- [ ] My existing card has incorrect data because of Department of Homeland Security (DHS) error. (Attach your existing card with incorrect data along with this application.)
- [ ] My name or other biographic information has been legally changed since issuance of my existing card.
- [X] My existing card has already expired or will expire within six months. (Typed)
- [ ] I have reached my 14th birthday and am registering as required. My existing card will expire after my 16th birthday. (See Note below for additional information.)
- [ ] I have reached my 14th birthday and am registering as required. My existing card will expire before my 16th birthday. (See Note below for additional information.)
- **NOTE:** If you are filing this application before your 14th birthday, or more than 30 days after your 14th birthday, you must select reason 2.a. Otherwise, if your card has expired, you must select reason 2.f.
- Section B:
- [ ] I am a permanent resident who is taking up commuter status.
- [ ] My Port-of-Entry (POE) into the United States will be (City or Town and State): __________
- [ ] I am a commuter who is taking up actual residence in the United States.
- Section C:
- [ ] I have been automatically converted to lawful permanent resident status.
- [ ] I have a prior edition of the Alien Registration Card, or am applying to replace my current Permanent Resident Card for a reason that is not specified above.
---
**Document Name:** Form I-90
**Page Number:** 1 of 7

### Extracted Information:

#### Part 1. Application Type (continued)
- **Section B (To be used only by a conditional permanent resident):**
- **1.a.** My previous card has been lost, stolen, or destroyed. [ ]
- **1.b.** My previous card was issued but never received. [ ]
- **1.c.** My existing card has been mutilated. [ ]
- **1.d.** My existing card has incorrect data because of DHS error. (Attach your existing permanent resident card with incorrect data along with this application.) [ ]
- **1.e.** My name or other biographic information has legally changed since the issuance of my existing card. [ ]

#### Part 3. Processing Information
- **1. Location where you applied for an immigrant visa or adjustment of status:**
- **Location:** [Handwritten/Typed]
- **2. Location where your immigrant visa was issued or USCIS office where you were granted adjustment of status:**
- **Location:** [Handwritten/Typed]
- **3. Destination in the United States at time of admission:**
- **Destination:** [Handwritten/Typed]
- **4. Port of Entry where admitted to the United States (City or Town and State):**
- **Port of Entry:** [Handwritten/Typed]
- **5.a.** Have you ever been in exclusion, deportation, or removal proceedings or ordered removed from the United States?
- [ ] Yes
- [ ] No
- **5.b.** Have you ever been granted permanent residence, had your card filed in 1-407, abandonment by Office of Return or United States Consulate, or otherwise been determined to have abandoned your status?
- [ ] Yes
- [ ] No

#### Part 4. Accommodations for Individuals with Disabilities and/or Impairments
- **1.** Are you requesting an accommodation because of your disabilities and/or impairments?
- [ ] Yes
- [ ] No
- **2.a.** I am deaf or hard of hearing and request the following accommodation (if you are requesting a sign language interpreter, indicate for which language):
- [Handwritten/Typed]

### Biographic Information
- **1. Ethnicity (Select only one box):**
- [ ] Hispanic or Latino
- [ ] Not Hispanic or Latino
- **2. Race (Select all applicable boxes):**
- [ ] White
- [ ] Asian
- [ ] Black or African American
- [ ] American Indian or Alaska Native
- [ ] Native Hawaiian or Other Pacific Islander
- **3. Height:**
- **Feet:** [Handwritten/Typed]
- **Inches:** [Handwritten/Typed]
- **4. Weight:**
- **Pounds:** [Handwritten/Typed]
- **5. Eye Color (Select only one box):**
- [ ] Black
- [ ] Blue
- [ ] Brown
- [ ] Gray
- [ ] Green
- [ ] Hazel
- [ ] Maroon
- [ ] Pink
- [ ] Unknown/Other
- **6. Hair Color (Select only one box):**
- [ ] Bald (No hair)
- [ ] Black
- [ ] Blond
- [ ] Brown
- [ ] Gray
- [ ] Red
- [ ] Sandy
- [ ] White
- [ ] Unknown/Other

### Instructions:
- **Part 4. Accommodations for Individuals with Disabilities and/or Impairments:**
- Read the information in the Form I-90 Instructions before completing this part.
- If you need extra space to complete this section, use the space provided in Part 8. Additional Information.
- If you answered "Yes," select any applicable boxes.

### Signature:
- No signature field or placeholder is present on this page.
---
**Document Name:** Form I-912

### Extracted Information:

#### Part 4. Accommodations for Individuals with Disabilities and/or Impairments (continued)
- **5.a.** [ ] I am blind or have low vision and request the following accommodation:
- **Accommodation Details:** (Handwritten/Typed: Not filled)

- **5.b.** [ ] I have another type of disability and/or impairment (Describe the nature of your disability and/or impairment and the accommodation you are requesting):
- **Description and Accommodation:** (Handwritten/Typed: Not filled)

#### Applicant's Contact Information
- **3. Applicant's Daytime Telephone Number:** (Handwritten/Typed: Not filled)
- **4. Applicant's Mobile Telephone Number (if any):** (Handwritten/Typed: Not filled)
- **5. Applicant's Email Address (if any):** (Handwritten/Typed: Not filled)

#### Applicant's Certification
- **Certification Text:**
- Copies of any documents I have submitted are exact photocopies of unaltered, original documents, and I understand that USCIS may require that I submit original documents at a later date. Furthermore, I authorize the release of any information from my records that USCIS may need to determine my eligibility for the benefit I seek.
- I further authorize release of information contained in this application, in supporting documents, and in USCIS records to other entities and persons where necessary for the administration and enforcement of U.S. immigration law.
- I understand that USCIS will require me to appear for an appointment to take my biometrics (fingerprints, photograph, and/or signature) and, at that time, I will be required to sign an oath reaffirming that:
1. I reviewed and provided or authorized all of the information in my application,
2. I understood all of the information contained in, and submitted with, my application, and
3. All of this information was complete, true, and correct at the time of filing.
- I certify, under penalty of perjury, that I provided or authorized all of the information in my application, I understood all of the information contained in, and submitted with, my application, and that all of this information is complete, true, and correct.

#### Applicant's Statement
- **NOTE:** Select the box for either Item Number 1.a. or 1.b. If applicable, select the box for Item Number 2.
- **1.a.** [ ] I can read and understand English, and I have read and understand every question and instruction on this application and my answer to every question.
- **1.b.** [ ] The interpreter named in Part 6. read to me every question and instruction on this application and my answer to every question in a language in which I am fluent and I understood everything.
- **Language:** (Handwritten/Typed: Not filled)
- **2.** [ ] At my request, the preparer named in Part 7., (Handwritten/Typed: Not filled), prepared this application for me based only upon information I provided or authorized.

#### Applicant's Signature
- **6.a. Applicant's Signature:** Signature present
- **6.b. Date of Signature (mm/dd/yyyy):** (Handwritten/Typed: Not filled)

#### Instructions:
- **NOTE TO ALL APPLICANTS:** If you do not complete 6.a. on this application or 6.b. or select required boxes as listed in the instructions, USCIS may deny your application.
---
**Document Name:** Form I-485
**Page Number:** 7 of 7

### Extracted Information:

#### Part 6. Interpreter's Contact Information, Certification, and Signature

- **Interpreter's Full Name:**
- 1.a. Interpreter's Family Name (Last Name): [Not filled]
- 1.b. Interpreter's Given Name (First Name): [Not filled]
- 1.c. Interpreter's Business or Organization Name: [Not filled]

- **Interpreter's Mailing Address:**
- 2.a. Street Number and Name: [Not filled]
- 2.b. Apt. Ste. Flr.: [Not filled]
- 2.c. City or Town: [Not filled]
- 2.d. State: [Not filled]
- 2.e. ZIP Code: [Not filled]
- 2.f. Province: [Not filled]
- 2.g. Postal Code: [Not filled]
- 2.h. Country: [Not filled]

- **Interpreter's Contact Information:**
- 3. Interpreter's Daytime Telephone Number: [Not filled]
- 4. Interpreter's Mobile Telephone Number (if any): [Not filled]
- 5. Interpreter's Email Address (if any): [Not filled]

- **Interpreter's Certification:**
- I certify, under penalty of perjury, that: [Not filled]
- I am fluent in English and [Not filled] which is the same language provided in Part 5., Item Number 1.b., and I have read to this applicant in the identified language every question and instruction on this application and his or her answer to every question. The applicant informed me that he or she understands every instruction, question, and answer on the application, including the Applicant's Certification, and has verified the accuracy of every answer. [Not filled]

- **Interpreter's Signature:**
- 6.a. Interpreter's Signature (sign in ink): [Not filled]
- 6.b. Date of Signature (mm/dd/yyyy): [Not filled]

#### Part 7. Contact Information, Declaration, and Signature of the Person Preparing this Application, if Other Than the Applicant

- **Preparer's Full Name:**
- 1.a. Preparer's Family Name (Last Name): SMITH (typed)
- 1.b. Preparer's Given Name (First Name): JOHN (typed)
- 1.c. Preparer's Business or Organization Name: [Not filled]

- **Preparer's Mailing Address:**
- 2.a. Street Number and Name: 1234 MAIN STREET (typed)
- 2.b. Apt. Ste. Flr.: [Not filled]
- 2.c. City or Town: SAN ANTONIO (typed)
- 2.d. State: TX (typed)
- 2.e. ZIP Code: 78201 (typed)
- 2.f. Province: [Not filled]
- 2.g. Postal Code: [Not filled]
- 2.h. Country: USA (typed)

- **Preparer's Contact Information:**
- 3. Preparer's Daytime Telephone Number: (210)555-1234 (typed)
- 4. Preparer's Mobile Telephone Number (if any): [Not filled]
- 5. Preparer's Email Address (if any): JOHN.SMITH@EMAIL.COM (typed)

### Instructions:

- Provide the following information about the interpreter.
- Provide the following information about the preparer.
- The interpreter must sign and date the form in ink.
- The preparer must provide their full name, mailing address, and contact information.
- The preparer must sign and date the form if they are not the applicant.

### Signature Fields:

- Interpreter's Signature: Not present
- Preparer's Signature: Not present
---
Document Name: Form I-485 Supplement J
Page Number: 4 of 7

**Extracted Information:**

**Preparer's Statement:**
- **1.a.** Checkbox selected: "I am not an attorney or accredited representative but have prepared this application on behalf of the applicant and with the applicant's consent." (Typed)

**Preparer's Certification:**
- The preparer certifies under penalty of perjury that they have reviewed the completed petition and that the information is complete, true, and correct. (Typed)

**Preparer's Signature:**
- **8.a.** Preparer's Signature: Signature present (Handwritten)
- **8.b.** Date of Signature: 12/15/2024 (Handwritten)

**Instructions:**
- If you are an attorney or accredited representative whose representation extends beyond preparation of this application, you may be obliged to submit a completed Form G-28, Notice of Entry of Appearance as Attorney or Accredited Representative, with this application.
---
**Document Name:** Form I-485 Supplement A

**Extracted Information:**

- **Part 8. Additional Information:**
- **Instructions:**
- If you need extra space to provide any additional information within this application, use the space below.
- If you need more space than what is provided, you may make copies of this page to complete and file with this application or attach a separate sheet of paper.
- Include your name and A-Number (if any) at the top of each sheet; indicate the Page Number, Part Number, and Item Number to which your answer refers, and sign and date each sheet.

- **Your Full Name:**
- **Family Name (Last Name):** MCCLAMROCK (Typed)
- **Given Name (First Name):** JAMES (Typed)
- **Middle Name:** ANDREW (Typed)
- **A-Number (if any):** A 0 0 0 0 0 0 0 0 0 (Typed)

- **Additional Information Sections:**
- **Page Number, Part Number, Item Number:** Multiple fields provided for additional information, all are blank.

**Signature Field:** No signature field is present on this page.
---
Document Name: Exhibit D


Information Extracted:
- The page contains only the text "EXHIBIT - D" which is typed.
- No other information or instructions are present on this page.
- No signature field or placeholder is present.
---
Document Name: Passport Copy

Extracted Information:
- Document Type: Passport
- Country: Jamaica
- Passport Number: Not visible
- Personal Information: Not visible
- Photograph: Present
- Signature: Not applicable (no signature field visible)

Instructions:
- "Copy of Biometrics Page of Passport"
- "Copy in Color to ensure that it is a quality copy"

Note: The information on the passport is not clearly visible in the image provided.
---
Document Name: Exhibit E

- This page is a cover page for Exhibit E.
- No information is filled out on this page.
- No instructions or signature fields are present.
---
Document Name: Copy of Front and Back of Green Card

Extracted Information:
- Document Type: United States of America Permanent Resident Card
- Name: Specimen
- Given Name: Test V
- USCIS Number: 000-000-000
- Country of Birth: Democratic Republic of Congo
- Category: IR1
- Resident Since: 08/16/11
- Card Expires: 08/16/21

Instructions:
- "Copy of Front and Back of Green Card"
- "(Copy in Color to ensure that it is a quality copy)"

Note: The information appears to be typed. There is no signature field or placeholder on this page.

"""

        try:
            # Convert image directly to bytes for API
            with Image.open(image_path) as image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert to base64
                encoded_image = base64.b64encode(img_byte_arr).decode('ascii')

            # Prepare chat prompt
            chat_prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "\n"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text", 
                            "text": "Please extract all information from this image, being especially careful with checkboxes and form fields. If you're unsure about any information, indicate that clearly."
                        }
                    ]
                }
            ]

            # Get completion with optimized parameters
            completion = self.retry_with_backoff(lambda: self.client.chat.completions.create(
                model=OCR_AZURE_DEPLOYMENT_NAME,
                messages=chat_prompt,
                temperature=0.0,  # Maximum consistency
                top_p=0.90,      # Slightly increased for better accuracy
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            ))

            return page_num, completion.choices[0].message.content

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {str(e)}")
            return page_num, f"Error processing page: {str(e)}"

    def combine_markdown_files(self, markdown_contents):
        """Combine all markdown content into a single string"""
        combined_content = ""
        
        for i, content in enumerate(markdown_contents):
            combined_content += f"## Page {i+1}\n\n"
            combined_content += content
            combined_content += "\n\n---\n\n"
        
        return combined_content
    
    def get_available_threads(self):
        """Calculate available threads based on CPU and memory usage."""
        total_cores = os.cpu_count() or 1  # Get total CPU cores, default to 1 if None
        free_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
        current_cpu_usage = psutil.cpu_percent(interval=1)  # Measure CPU usage over 1 sec

        # Adjust max threads based on CPU and memory
        if current_cpu_usage > 80:  # If CPU is heavily loaded, reduce threads
            max_threads = max(2, total_cores // 2)
        elif free_memory < 1:  # If available memory is low, reduce threads
            max_threads = max(2, total_cores // 2)
        else:
            max_threads = total_cores * 2  # Default: Allow 2x CPU cores for I/O tasks

        return max_threads

    def convert_pdf(self, pdf_content: bytes):
        """Main conversion process"""
        try:
            # Split PDF into images
            print("Converting PDF pages to images...")
            image_paths = self.split_pdf_to_images(pdf_content)
            
            # Convert each image to markdown using thread pool
            print("Converting images to markdown using parallel processing...")
            markdown_contents = [None] * len(image_paths)  # Pre-allocate list to maintain order
            
            # Create list of tuples with (image_path, page_number)
            image_tasks = [(path, idx) for idx, path in enumerate(image_paths)]
            
            # Process images in parallel with max threads
            max_threads = min(self.get_available_threads(), len(image_tasks), MAX_THREADS)
            print(f">>>> Using {max_threads} threads.")

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Submit all tasks and create future-to-index mapping
                future_to_page = {
                    executor.submit(self.image_to_markdown, task): task[1]
                    for task in image_tasks
                }
                
                # Process completed futures and store results in order
                for future in as_completed(future_to_page):
                    page_num, content = future.result()
                    markdown_contents[page_num] = content
            
            # Combine all markdown content
            print("Combining markdown content...")
            final_content = self.combine_markdown_files(markdown_contents)

            if SAVE_TO_MARKDOWN:
                # Save markdown content to file
                with open("markdown_gpt.md", 'w', encoding='utf-8') as f:
                    f.write(final_content)
            
            # Cleanup temporary files and directories
            for image_path in image_paths:
                os.remove(image_path)
            os.removedirs(self.temp_dir)
                
            print(f"Conversion complete!")
            return final_content
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # Cleanup on error
            for image_path in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
            if os.path.exists(self.temp_dir):
                os.removedirs(self.temp_dir)
            raise

class ConverterByDocumentIntelligence:
    def format_with_openai(self, markdown_content):
        client = AzureOpenAI(
            api_key=DI_AZURE_OPENAI_KEY,
            api_version=DI_AZURE_OPENAI_API_VERSION,
            azure_endpoint=DI_AZURE_OPENAI_ENDPOINT
        )

        prompt = """Please reformat this form content into clear, well-structured markdown. 
        Requirements:
        1. Preserve all form fields, instructions, and text
        2. Show all checkboxes ([X] for selected, [ ] for unselected)
        3. Maintain proper spacing and hierarchy
        4. Keep section numbering and titles
        5. Format instructions in italics
        6. Group related fields together
        7. Make it highly readable
        8. INCLUDE EVERYTHING FROM THE ORIGINAL MARKDOWN - all check boxes and info should be included, even fields that were not filled out should be present.
        9. include the page numbers on every top page.

        Original form content:
        """

        try:
            response = client.chat.completions.create(
                model=DI_AZURE_DEPLOYMENT_NAME,  # o3-mini deployment
                messages=[
                    {
                        "role": "user",
                        "content": prompt + markdown_content
                    }
                ],
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure OpenAI: {str(e)}")
            return markdown_content  # Return original content if formatting fails

    def convert_pdf(self, pdf_content: bytes):
        try:
            # Initialize the client
            document_client = DocumentIntelligenceClient(
                endpoint=AZURE_DOCUMENT_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DOCUMENT_KEY)
            )

            print("Begin analyzing document using Document Intelligence...")

            # Split the PDF into chunks of 15 pages
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages)
            markdown_contents = []

            for start_page in range(0, total_pages, CHUNK_SIZE):
                end_page = min(start_page + CHUNK_SIZE, total_pages)
                print(f"Processing pages {start_page + 1} to {end_page}...")

                # Extract pages for the current chunk
                pdf_writer = PdfWriter()
                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf_reader.pages[page_num])

                # Convert the chunk to bytes
                chunk_bytes = io.BytesIO()
                pdf_writer.write(chunk_bytes)
                chunk_bytes.seek(0)

                # Process the chunk
                poller = document_client.begin_analyze_document(
                    "prebuilt-layout",
                    body=chunk_bytes.getvalue(),
                    content_type="application/pdf",
                    output_content_format=DocumentContentFormat.MARKDOWN
                )

                result = poller.result()
                markdown_contents.append(result.content)

            # Combine all markdown content
            combined_markdown = "\n\n---\n\n".join(markdown_contents)

            if FORMAT_MARKDOWN_FROM_DI:
                print("Formatting markdown started with OpenAI...")
                # Format the markdown with OpenAI
                formatted_markdown = self.format_with_openai(combined_markdown)
                print("Formatting markdown completed with OpenAI...")
            else:
                formatted_markdown = combined_markdown

            if SAVE_TO_MARKDOWN:
                # Save both original and formatted markdown
                with open(f"markdown_di_raw.md", "w", encoding="utf-8") as f:
                    f.write(combined_markdown)

                if FORMAT_MARKDOWN_FROM_DI:
                    with open(f"markdown_di_formatted.md", "w", encoding="utf-8") as f:
                        f.write(formatted_markdown)

            print(f"Document Intelligence parsing complete!")
            return formatted_markdown
        except Exception as e:
            print(f"An error occurred while parsing the document with Document Intelligence: {str(e)}")
            raise