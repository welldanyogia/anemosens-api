# Anemosense API

Flask API for anemia prediction using a TensorFlow model. This repository
includes Docker configurations for both development and production usage and
basic instructions for deploying on a VPS with SSL using Certbot.

## Development

Build the development image:

```bash
docker build -f Dockerfile.dev -t anemosense-dev .
```

Run it mounting the source code so changes are reflected immediately:

```bash
docker run --rm -it -p 5000:5000 -v $(pwd):/app anemosense-dev
```

The API will be available at `http://localhost:5000`.

## Production Build

Create the production image:

```bash
docker build -t anemosense-api .
```

Run the container in detached mode:

```bash
docker run -d --name anemosense-api -p 5000:5000 anemosense-api
```

Gunicorn is used as the WSGI server inside the container.

## Deploying on a VPS

1. **Install Docker & Docker Compose**

   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo systemctl enable --now docker
   ```

2. **Clone this repository on the server**

   ```bash
   git clone <repo-url>
   cd anemosens-api
   ```

3. **Build and run the production container**

   ```bash
   docker build -t anemosense-api .
   docker run -d --name anemosense-api -p 5000:5000 anemosense-api
   ```

4. **Install Nginx and configure it as a reverse proxy**

   ```bash
   sudo apt install nginx -y
   ```

   Create `/etc/nginx/sites-available/anemosense` with the following contents:

   ```nginx
   server {
       listen 80;
       server_name anemosense.webranastore.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable the site and reload Nginx:

   ```bash
   sudo ln -s /etc/nginx/sites-available/anemosense /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```

5. **Obtain SSL certificates using Certbot**

   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot --nginx -d anemosense.webranastore.com
   ```

   Follow the prompts to obtain and install the certificate. Certbot will also
   create a cron job for automatic renewal.

After these steps the API will be served at
`https://anemosense.webranastore.com` via Nginx with HTTPS.
