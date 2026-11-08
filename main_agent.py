"""
--------
Terminal chat with the MATZ assistant, for local testing.
Run from Backend/:  python main_agent.py [organization_id]
Searches only the given organization's documents (default: matz-demo-org).
"""

import sys

from agents.langgraph_agent.main_langgraph_agent import chat


def main():
    print("MATZ Assistant (type 'exit' to quit)\n")
    history = []
    org_id = sys.argv[1] if len(sys.argv) > 1 else "matz-demo-org"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        answer, citations, history, _ = chat(user_input, history, organization_id=org_id)
        print(f"Matz: {answer}")

        # Print citations if any
        if citations:
            print("\n  Sources:")
            for c in citations:
                print(
                    f"  [{c['citation_order']}] {c['document_title']} · "
                    f"{c['collection_name']} · Page {c['page_number']} "
                    f"(relevance: {c['relevance_score']})"
                )
        print()


if __name__ == "__main__":
    main()