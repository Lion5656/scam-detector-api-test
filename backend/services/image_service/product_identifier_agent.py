from langchain_core.tools import tool
from langchain_groq import ChatGroq
from backend.config import settings

def create_product_identifier_agent():
    llm = ChatGroq(
        model="llama3-70b-8192", 
        temperature=0.1,  
        max_tokens=256
    )

    tools = [search_product_info, google_custom_search_product]
    
    # 給它一個簡單的 zero-shot / function calling chain，現在叫 agent
    # 因為 `initialize_agent` 被棄用，我們可以用 `bind_tools` 達到類似效果，
    # 但為了快速測試，我們直接回傳一個 callable function

    def agent_runner(prompt: str) -> str:
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(prompt)
        
        # Tool execution loop
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "search_product_info":
                    query = tool_call["args"].get("query", "")
                    res = search_product_info.invoke({"query": query})
                    tool_results.append(f"Search Result for '{query}':\n{res}")
                elif tool_call["name"] == "google_custom_search_product":
                    query = tool_call["args"].get("query", "")
                    res = google_custom_search_product.invoke({"query": query})
                    tool_results.append(f"Search Result for '{query}':\n{res}")
            
            # 再跑一次
            final_prompt = prompt + "\n\nTool Results:\n" + "\n".join(tool_results)
            final_response = llm.invoke(final_prompt)
            return final_response.content
            
        return response.content

    # 提供相同介面給 image_analyzer
    class MockAgent:
        def run(self, prompt: str) -> str:
            return agent_runner(prompt)

    return MockAgent()

@tool
def google_custom_search_product(query: str) -> str:
    """如果商品較為冷門，使用 Google Custom Search 進行商品查詢"""
    try:
        from googleapiclient.discovery import build
        API_KEY = getattr(settings, "GOOGLE_API_KEY", "") 
        CX = getattr(settings, "GOOGLE_CSE_ID", "") 
        
        if not CX or not API_KEY:
            return "無法執行 Google Custom Search，請確認設定 (API_KEY 或 CX 遺失)。"
            
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=CX, num=3).execute()
        
        items = res.get('items', [])
        if not items:
            return "搜尋不到結果"
            
        summary = []
        for item in items:
            summary.append(f"標題: {item.get('title')}")
            summary.append(f"摘要: {item.get('snippet')}")
        return "\n".join(summary)
    except Exception as e:
        return f"Google Search Error: {e}"

@tool
def search_product_info(query: str) -> str:
    """使用 DuckDuckGo 搜尋商品資訊"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "找不到相關商品資訊。"
            
        # 整理搜尋結果
        summary = []
        for r in results:
            summary.append(f"標題: {r.get('title', '')}")
            summary.append(f"內容摘要: {r.get('body', '')}")
            summary.append("---")
            
        return "\n".join(summary)
        
    except Exception as e:
        return f"搜尋時發生錯誤: {str(e)}"
