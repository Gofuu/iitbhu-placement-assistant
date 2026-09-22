"""
Single entrypoint that runs the whole ingestion pipeline in order:

  1. parse_policy   -> data/processed/policy/*.md (+ .meta.json)
  2. build_sql_db   -> data/placement.db  (recruiter_records table)
  3. build_vectorstore -> data/vectorstore/ (Chroma, policy + forum chunks)

Run: python -m src.ingest.run_ingestion
"""
from src.ingest import parse_policy, build_sql_db, build_vectorstore


def main():
    print("=== 1/3 Parsing policy documents ===")
    parse_policy.main()

    print("\n=== 2/3 Building SQLite recruiter DB ===")
    build_sql_db.main()

    print("\n=== 3/3 Building vector store (policy + forum) ===")
    build_vectorstore.main()

    print("\nIngestion complete. You now have:")
    print("  data/processed/policy/  -- parsed policy markdown")
    print("  data/placement.db       -- recruiter_records SQL table")
    print("  data/vectorstore/       -- Chroma vector store (policy + forum)")


if __name__ == "__main__":
    main()
