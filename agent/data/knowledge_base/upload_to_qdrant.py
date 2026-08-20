"""
data/knowledge_base/upload_to_qdrant.py
----------------------------------------
One-time script to embed the mock knowledge base and upload to Qdrant Cloud.
Run this once to populate the matz_chunks collection.

Also creates all required payload indexes so filtering works correctly:
  - organization_id → used in every search query (tenant isolation)
  - document_id     → used in delete document
  - collection_id   → used in collection-scoped search

Usage:
    cd langgraph_agent
    python data/knowledge_base/upload_to_qdrant.py
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.embeddings import embed_texts, EMBED_DIMENSIONS

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.environ.get("QDRANT_URL")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY")
COLLECTION      = "matz_chunks"
ORGANIZATION_ID = "matz-demo-org"

# ── Mock KB ───────────────────────────────────────────────────────────────────
MOCK_KB = [
    {
        "chunk_id":        "chunk-001",
        "document_id":     "doc-001",
        "document_title":  "Remote Work Guidelines",
        "collection_id":   "col-hr-policies",
        "collection_name": "HR Policies",
        "page_number":     2,
        "chunk_index":     0,
        "content": (
            "Remote Work Policy (HR Policies · Section 2.1): "
            "Employees may work remotely up to three days per week with manager approval. "
            "Core collaboration hours are 10:00 AM to 3:00 PM in the employee's local time zone. "
            "Employees are responsible for maintaining a secure, productive workspace. "
            "A stable internet connection of at least 10 Mbps is required."
        ),
    },
    {
        "chunk_id":        "chunk-002",
        "document_id":     "doc-002",
        "document_title":  "Employee Handbook 2024",
        "collection_id":   "col-hr-policies",
        "collection_name": "HR Policies",
        "page_number":     18,
        "chunk_index":     0,
        "content": (
            "Leave Policy (HR Policies · Section 3.2): "
            "Full-time employees are entitled to 20 days of annual leave per year. "
            "Leave must be requested at least 5 working days in advance via the HR portal. "
            "Unused leave can be carried forward up to a maximum of 10 days to the next year. "
            "Sick leave is separate and capped at 10 days per year with a medical certificate required after 3 consecutive days."
        ),
    },
    {
        "chunk_id":        "chunk-003",
        "document_id":     "doc-003",
        "document_title":  "Information Security Policy",
        "collection_id":   "col-security",
        "collection_name": "Security",
        "page_number":     5,
        "chunk_index":     0,
        "content": (
            "Information Security Policy (Security · Section 4.1): "
            "All employees must use strong passwords of at least 12 characters including uppercase, "
            "lowercase, numbers, and special characters. Passwords must be changed every 90 days. "
            "Multi-factor authentication (MFA) is mandatory for all company systems. "
            "Never share your credentials with anyone, including IT staff."
        ),
    },
    {
        "chunk_id":        "chunk-004",
        "document_id":     "doc-002",
        "document_title":  "Employee Handbook 2024",
        "collection_id":   "col-hr-policies",
        "collection_name": "HR Policies",
        "page_number":     4,
        "chunk_index":     1,
        "content": (
            "Onboarding Process (HR Policies · Section 1.1): "
            "New employees should complete their onboarding checklist within the first 5 working days. "
            "This includes setting up company accounts, completing compliance training, and meeting with "
            "their line manager. IT equipment will be provided on day one. "
            "Buddy system is in place — each new hire is assigned an onboarding buddy for the first 30 days."
        ),
    },
    {
        "chunk_id":        "chunk-005",
        "document_id":     "doc-004",
        "document_title":  "Customer Support Playbook",
        "collection_id":   "col-operations",
        "collection_name": "Operations",
        "page_number":     12,
        "chunk_index":     0,
        "content": (
            "Expense Policy (Operations · Section 6.3): "
            "All business expenses must be submitted within 30 days of being incurred. "
            "Receipts are required for any expense above $25. "
            "Travel expenses including flights and hotels require pre-approval from your manager. "
            "Meal allowance is capped at $50 per day during business travel. "
            "Submit claims through the company expense portal."
        ),
    },
    {
        "chunk_id":        "chunk-006",
        "document_id":     "doc-005",
        "document_title":  "Vendor Risk Assessment",
        "collection_id":   "col-compliance",
        "collection_name": "Compliance",
        "page_number":     3,
        "chunk_index":     0,
        "content": (
            "Vendor Management Policy (Compliance · Section 2.1): "
            "All new vendor relationships must go through the procurement team for approval. "
            "Vendors handling company data must sign a Data Processing Agreement (DPA) before engagement. "
            "Vendor risk assessments are conducted annually for all tier-1 suppliers. "
            "Purchase orders above $10,000 require CFO approval."
        ),
    },
    {
        "chunk_id":        "chunk-007",
        "document_id":     "doc-006",
        "document_title":  "Q3 Sales Enablement Deck",
        "collection_id":   "col-sales",
        "collection_name": "Sales Enablement",
        "page_number":     8,
        "chunk_index":     0,
        "content": (
            "Sales Guidelines (Sales Enablement · Section 1.2): "
            "Standard discounts are capped at 15% without VP Sales approval. "
            "All proposals must be reviewed by the legal team before being sent to enterprise clients. "
            "Quote validity period is 30 days from the date of issue. "
            "Use the approved proposal template in the Sales Enablement collection."
        ),
    },
    {
        "chunk_id":        "chunk-008",
        "document_id":     "doc-007",
        "document_title":  "Brand Voice & Messaging",
        "collection_id":   "col-marketing",
        "collection_name": "Marketing",
        "page_number":     1,
        "chunk_index":     0,
        "content": (
            "Brand Guidelines (Marketing · Section 1.1): "
            "The primary brand color is #185FA5 (MATZ Blue). "
            "Always use the approved logo files — never stretch, recolor, or modify the logo. "
            "The approved typeface is Inter for all digital materials. "
            "All external marketing materials must be reviewed by the brand team before publishing."
        ),
    },
]


def main():
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Delete and recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' already exists — deleting and recreating...")
        client.delete_collection(COLLECTION)

    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIMENSIONS, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' created.")

    # ── Create payload indexes ────────────────────────────────────────────────
    # Required for filtering — without these, Qdrant returns 400 errors
    print("Creating payload indexes...")

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="organization_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("  ✅ organization_id index created (used for tenant isolation in every search)")

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("  ✅ document_id index created (used for delete document)")

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="collection_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("  ✅ collection_id index created (used for collection-scoped search)")

    # ── Embed and upload chunks ───────────────────────────────────────────────
    print(f"\nEmbedding {len(MOCK_KB)} chunks...")
    contents = [chunk["content"] for chunk in MOCK_KB]
    vectors  = embed_texts(contents)

    points = []
    for i, (chunk, vector) in enumerate(zip(MOCK_KB, vectors)):
        print(f"  [{i+1}/{len(MOCK_KB)}] {chunk['document_title']} — page {chunk['page_number']}")
        points.append(PointStruct(
            id=i + 1,
            vector=vector,
            payload={
                "chunk_id":        chunk["chunk_id"],
                "document_id":     chunk["document_id"],
                "document_title":  chunk["document_title"],
                "collection_id":   chunk["collection_id"],
                "collection_name": chunk["collection_name"],
                "page_number":     chunk["page_number"],
                "chunk_index":     chunk["chunk_index"],
                "content":         chunk["content"],
                "organization_id": ORGANIZATION_ID,
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)

    info = client.get_collection(COLLECTION)
    print(f"\n✅ Successfully uploaded {len(points)} chunks to Qdrant Cloud.")
    print(f"   Collection:   {COLLECTION}")
    print(f"   Organization: {ORGANIZATION_ID}")
    print(f"   Total vectors: {info.points_count}")
    print(f"   Indexes: organization_id, document_id, collection_id")


if __name__ == "__main__":
    main()