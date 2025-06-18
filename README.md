# Instructions for setting up the development environment
## Prerequisits
1. Install [Git](https://git-scm.com/downloads)
2. Install [Docker Desktop](https://docs.docker.com/desktop/)
## Cloning the repository
### Using SSH authentication (recommended)
1. Create ssh key ([Tutorial for all plattforms](https://www.digitalocean.com/community/tutorials/how-to-create-ssh-keys-with-openssh-on-macos-or-linux))
2. Add public key to github in: Settings -> SSH and GPG keys
3. In a terminal, cd to the preferred directory and type: `git clone git@github.com:PSE-Hyperion/music-generator.git`
### Using HTTP
1. In a terminal, cd to the preferred directory and type: `git clone https://github.com/PSE-Hyperion/music-generator.git`
2. Log in with your github account
## Setting up the docker container
`docker pull tensorflow/tensorflow`
