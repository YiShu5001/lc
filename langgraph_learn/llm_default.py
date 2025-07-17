OPENAI_API_KEY="sk-3GiWozbrq1kFipcgF7ymSbgH9g1f5lWGfpkUgfWBICul0Kai"
OPENAI_API_BASE_URL="https://www.dmxapi.cn/v1"
from langchain_openai import ChatOpenAI
def getDefaultOPENAI() -> ChatOpenAI:

    # 如果有初始状态，就创建一个回调 handler
    # callbacks = []
    # if initial_state is not None:
    #     callbacks.append(ThoughtCaptureHandler(initial_state))
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE_URL,
        timeout=120,
        # streaming=True,  # 关键：streaming 为 True 才会逐 token 触发 on_llm_new_token
        # callbacks=callbacks
    )
    return llm


