Set objShell = CreateObject("Wscript.Shell")
objShell.Run "curl.exe -s -o NUL https://quantmatrix-ai-api.onrender.com/health", 0, False
