"""" Methods of processing data """
import os
import time


def remove_data_files(file_names=['sqlite.db', 'faiss.index']):
    """
    delete files
    :param file_names: file name list
    :return: None
    """
    for file in file_names:
        if os.path.isfile(file):
            os.remove(file)


def log_time_func(func_name, delta_time):
    """
    print function time
    :param func_name: function name
    :param delta_time: consumed time
    :return: None
    """
    log.info("func `{}` consume time: {:.2f}s".format(func_name, delta_time))


def disable_cache(*args, **kwargs):
    """
    disable cache
    """
    return False

def mock_chat_completion(*args, **kwargs):
    """Mock LLM: returns a simple echo-style response instantly."""
    messages = kwargs.get("messages", [])
    user_content = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            user_content = m.get("content", "")
            break
    content = f"[MOCK] Answer to: {user_content}" if user_content else "[MOCK] Hello."
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }
