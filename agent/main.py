"""
main.py
--------
Entry point for the MATZ AI Knowledge Assistant.
Run: python main.py
"""

from agent.src.agent.graph import chat


def main():
    print("MATZ Assistant (type 'exit' to quit)\n")
    history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        answer, citations, history = chat(user_input, history)
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