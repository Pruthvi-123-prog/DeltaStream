import os
import time

try:
    from deltastream.runtime import DeltaStreamRuntime
    model = DeltaStreamRuntime("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "delta_tinyllama_1.1b_chat_v1.0")
    print("\n[*] Model initialized.")
    
    # We will modify runtime.py temporarily to not swallow the exception
    # Or just let it print
    
    result = model.generate("hello")
    print(f"\nType of result: {type(result)}")
    print(f"\nResult[0]:\n{result[0]}")
    print(f"\nResult[1] (stats):\n{result[1]}")
except Exception as e:
    import traceback
    traceback.print_exc()
