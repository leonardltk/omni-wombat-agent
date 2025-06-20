# Omni-Wombat-Agent
<img src="images/whimsical_wombat.png" width="300px" alt="Whimsical Wombat">

## 1. Host the App Servers
```bash
python MCP_server_grab.py --port 7860 &
python MCP_server_gojek.py --port 7861 &
python MCP_server_RedMart.py --port 7862 &
```

## 2. Host the chat client
```bash
python mistral_client_remote.py
```

## 3. Host the ASR client
```bash
python ASR_OpenAI_client_remote.py
# or
python ASR_Gemini_client_remote.py
```
