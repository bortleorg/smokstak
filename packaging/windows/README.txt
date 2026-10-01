smokstak - Windows x64 portable build

1. Extract the entire ZIP to a writable folder on your Windows 10/11 machine.
2. Double-click Start-smokstak.cmd.
3. Your default browser opens the local stacking page. Select source data on
   this machine and choose an output folder with sufficient free disk space.
4. Keep the launcher window open while processing. Close it to stop the app.

No Rust, Python, Node.js, CUDA toolkit, or separate web server is required.
The C runtime is statically linked. The browser UI is embedded in the executable.
The server listens only on this machine (normally http://127.0.0.1:7878).
If the port is occupied, it tries subsequent ports and prints the actual URL.
Open the browser on the machine running smokstak; it is not a LAN web service.

For a manual launch from PowerShell:
  .\smokstak.exe gui
For all command-line options:
  .\smokstak.exe --help

The reconstruction engine currently uses CPU worker threads and system RAM.
An RTX 4090 is compatible with this build but is not used for acceleration.
Large astronomical datasets can require substantial RAM, disk space, and time.

These are development builds.
BUILD.txt identifies the exact source revision and Rust compiler used.
