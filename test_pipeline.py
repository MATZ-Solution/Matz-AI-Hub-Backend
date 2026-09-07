"""
test_pipeline.py
-----------------
Test script for the document processing pipeline.
Creates a sample company policy PDF and runs it through the full pipeline.

Run:
    cd Agents/langgraph_agent
    python test_pipeline.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.pipeline.processor import extract_text
from data.pipeline.chunker import chunk_text
from data.pipeline.ingestor import ingest_chunks
from agents.langgraph_agent.utils.utils import logger


def create_test_pdf():
    """Create a simple test PDF with company policy content."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        has_reportlab = True
    except ImportError:
        has_reportlab = False

    if has_reportlab:
        pdf_path = "test_document.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "MATZ Company - IT Support Policy")
        c.setFont("Helvetica", 12)
        y = 700
        lines = [
            "IT Support Policy (IT Department · Section 5.1):",
            "",
            "All IT support requests must be submitted through the company helpdesk portal.",
            "Response times are as follows:",
            "- Critical issues (system down): 1 hour response time",
            "- High priority (major feature broken): 4 hour response time",
            "- Medium priority (minor issues): 1 business day",
            "- Low priority (general questions): 3 business days",
            "",
            "Employees must not attempt to fix hardware issues themselves.",
            "All company laptops must be encrypted using BitLocker or FileVault.",
            "Software installations require IT department approval.",
            "Personal devices must not be connected to the company network",
            "without prior approval from the IT security team.",
            "",
            "IT Equipment Policy (IT Department · Section 5.2):",
            "",
            "Company equipment must be returned upon resignation or termination.",
            "Equipment damage due to negligence may result in repair cost deduction.",
            "Employees are responsible for keeping their devices updated.",
            "All devices must have antivirus software installed and active.",
        ]
        for line in lines:
            c.drawString(100, y, line)
            y -= 20
            if y < 100:
                c.showPage()
                y = 750
        c.save()
        logger.info("Test PDF created: %s", pdf_path)
        return pdf_path
    else:
        # Create a text file instead
        txt_path = "test_document.txt"
        with open(txt_path, "w") as f:
            f.write("""IT Support Policy (IT Department · Section 5.1):

All IT support requests must be submitted through the company helpdesk portal.
Response times are as follows:
- Critical issues (system down): 1 hour response time
- High priority (major feature broken): 4 hour response time
- Medium priority (minor issues): 1 business day
- Low priority (general questions): 3 business days

Employees must not attempt to fix hardware issues themselves.
All company laptops must be encrypted using BitLocker or FileVault.
Software installations require IT department approval.
Personal devices must not be connected to the company network
without prior approval from the IT security team.

IT Equipment Policy (IT Department · Section 5.2):

Company equipment must be returned upon resignation or termination.
Equipment damage due to negligence may result in repair cost deduction.
Employees are responsible for keeping their devices updated.
All devices must have antivirus software installed and active.
""")
        logger.info("Test TXT created (reportlab not installed): %s", txt_path)
        return txt_path


def main():
    print("\n" + "="*60)
    print("MATZ Document Processing Pipeline Test")
    print("="*60 + "\n")

    # Step 1 — Create test document
    print("Step 1: Creating test document...")
    file_path = create_test_pdf()
    print(f"  ✅ Test document created: {file_path}\n")

    # Step 2 — Extract text
    print("Step 2: Extracting text...")
    result = extract_text(file_path)

    if not result["success"]:
        print(f"  ❌ Extraction failed: {result['error']}")
        return

    print(f"  ✅ Text extracted using: {result['method']}")
    print(f"  ✅ Characters: {len(result['text'])}")
    print(f"  ✅ Pages: {result['page_count']}")
    print(f"  Preview: {result['text'][:100]}...\n")

    # Step 3 — Chunk text
    print("Step 3: Chunking text...")
    document_id = "doc-test-it-policy"
    chunks = chunk_text(
        text=result["text"],
        document_id=document_id,
        page_count=result["page_count"]
    )
    print(f"  ✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['token_count']} tokens | page {chunk['page_number']} | {chunk['content'][:60]}...")
    print()

    # Step 4 — Ingest to Qdrant
    print("Step 4: Uploading to Qdrant...")
    vector_ids = ingest_chunks(
        chunks=chunks,
        document_id=document_id,
        document_title="IT Support Policy",
        collection_id="col-it-support",
        collection_name="IT Support",
        organization_id="matz-demo-org",
    )
    print(f"  ✅ Uploaded {len(vector_ids)} chunks to Qdrant")
    print(f"  Vector IDs: {vector_ids[:2]}...\n")

    print("="*60)
    print("✅ Pipeline complete! Now ask the agent about IT support.")
    print("   Try: 'what is the IT support response time?'")
    print("="*60 + "\n")

    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)


if __name__ == "__main__":
    main()