This tool dump the database generated from JFA GO and uses the collected information to display a popup notification with the expiry date for every organizr user if logged in

to identify the user organizr v2 is needed 

![Beispiel](https://github.com/baervers23/expiry-notifier/blob/main/img/user.png)

# Installation

1 - git clone https://github.com/baervers23/expiry-notifier.git

2 - Edit .env file

3 - Edit docker-compose.yml file

4 - Build (prefer clean/no-cache)

5 - Run with docker compose


# Commands

sudo docker compose build                        //build

sudo docker compose build --no-cache             //build clean

sudo docker compose up -d                        //run

# Organizr Popup

For the Organizr V2 Widget just add the content of "customJS" in the Organizr V2 CustomJavaScript section.


# Troubleshoot API

- healthcheck                               ->      /health
  
- organizr membercheck (group, user, jwt)   ->      /api/v1/membership

