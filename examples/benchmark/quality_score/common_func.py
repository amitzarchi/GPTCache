"""" Methods of processing data """
import os
import time


def remove_data_files(file_extension, file_path):
    """
    delete files
    :param file_names: file name list
    :return: None
    """
    file_names = [os.path.join(file_path, f'sqlite_{file_extension}.db'), os.path.join(file_path, f'faiss_{file_extension}.index')]
    for file in file_names:
        if os.path.isfile(file):
            os.remove(file)


def ensure_directory_permissions(file_path):
    """
    Ensure the directory exists and has proper permissions
    :param file_path: directory path
    :return: None
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    # Ensure directory has proper permissions
    os.chmod(file_path, 0o755)


def ensure_file_permissions(file_path):
    """
    Ensure the file has proper read/write permissions
    :param file_path: file path
    :return: None
    """
    if os.path.exists(file_path):
        os.chmod(file_path, 0o664)


def mock_chat_completion(*args, **kwargs):
    """Mock LLM: returns a simple echo-style response instantly."""
    messages = kwargs.get("messages", [])
    user_content = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            user_content = m.get("content", "")
            break
    content = user_content if user_content else "[MOCK] Hello."
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
