# Setup Guide for Windows using Fedora Linux

## Fedora Linux Installation
1. Install Fedora for WSL by running `wsl --install -d Fedora` in cmd.
2. Enter a username (and password if you want).
3. Close the cmd window, reopen it, click the dropdown arrow in the terminal, and select Fedora. You are now in the Fedora (Linux) terminal.

## SSH key in Fedora terminal
1. Generate an SSH key by running `ssh-keygen`, press Enter to accept defaults (and optionally set a password).
2. Then copy your public key with `cat ~/.ssh/id_ed25519.pub` (use Tab for autocomplete if necessary).
3. Copy the output, which should look like `ssh-ed25519 ... username@device`.
4. Go to GitHub, open your profile settings, then SSH and GPG keys, click New SSH key, choose any title, leave type as Authentication Key, paste the copied SSH key into the Key field, and click Add SSH key.

## Cloning the Repository
1. Next, go to the GitHub repository, click the green Code button, select SSH, and copy the link.
2. Back in the Fedora terminal, run `eval $(ssh-agent)` and then `ssh-add`, followed by `sudo dnf install git`.
3. Clone the repository with `git clone [your-copied-SSH-link]`. If asked to confirm authenticity, type yes.
4. Move into the project directory with `cd music-generator/` and switch to the correct branch using `git checkout main`.
5. Then go into the scripts folder and run the setup scripts one after the other: `cd scripts/`, `./install_on_fedora.sh`, and `./setup.sh`.

## VSCode
1. Now open VS Code. Make sure the WSL extension is installed (blue background with a penguin icon).
2. Press F1 and select “WSL: Connect to WSL using Distro...” and then choose Fedora.
3. Open the folder `music-generator/` and if asked whether you trust the workspace, select Yes.
4. You can now run the project by clicking Run → Start Debugging.
5. If errors appear due to missing extensions, install them inside WSL (not locally). The required extensions are Python, Ruff, and EditorConfig for VS Code (the one with the mouse and glasses icon). VS Code may also recommend them automatically — just accept and install. Once these are installed, everything should work and you can run and debug the project in Fedora WSL through VS Code.

## Dropbox
The project uses Dropbox to share generated data between the developers. If you aren't part of the team, you can run the program and the needed folders will be generated automatically.
