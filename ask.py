import requests
import json
import sys

def ask_question(question: str):
    print(f"🤔 Asking: '{question}'\n")
    
    # Hit the /query endpoint we just built
    url = "http://localhost:8000/query"
    payload = {"question": question}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("✅ ANSWER:")
        print(data["answer"])
        print("\n" + "-"*50)
        
        print("\n🔍 PIPELINE TRACE:")
        print(f"Rewritten Query: {data['rewritten_query']}")
        print("\nSources Used:")
        for source in data["sources"]:
            print(f"  - {source}")
            
        print("\nDocument Grades:")
        for doc in data["doc_grades"]:
            grade_icon = "🟢" if doc["grade"] == "yes" else "🔴"
            print(f"  {grade_icon} {doc['source']}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Is it running?")
        print("Run: uvicorn app.main:app --reload --port 8000")

if __name__ == "__main__":
    # You can pass a question as a command line argument, e.g.:
    # python ask.py "What is the holding period for a short term capital asset?"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        # Default question if none provided
        q = "If I hold a listed equity share for 10 months, is it a short-term capital asset?"
        
    ask_question(q)
