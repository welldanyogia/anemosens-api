
# Deployment Guide (Docker + Nginx + SSL)

## Prerequisites
1. Docker and Docker Compose installed on the server.
2. Domain `anemosense.webrana.id` pointing to the server IP.
3. Ports 80 and 443 open.

## Initial Setup (Run on Server)
1. Clone the repository.
2. Edit `init-letsencrypt.sh`:
   - Update `email` variable.
3. Make the script executable:
   ```bash
   chmod +x init-letsencrypt.sh
   ```
4. Create `.env` file from example:
   ```bash
   cp .env.example .env
   nano .env
   # Edit any values if necessary (e.g., SECRET_KEY)
   ```
5. Run the initialization script to generate SSL certificates:
   ```bash
   ./init-letsencrypt.sh
   ```
5. Once certificates are generated, the API will be accessible at https://anemosense.webrana.id.

## CI/CD Setup (GitHub Actions)
1. Go to your GitHub repository > Settings > Secrets.
2. Add the following secrets:
   - `DOCKERHUB_USERNAME`: Your DockerHub username.
   - `DOCKERHUB_TOKEN`: Your DockerHub access token.
   - `VPS_HOST`: Server IP address.
   - `VPS_USERNAME`: Server SSH username (e.g., `root` or `ubuntu`).
   - `VPS_SSH_KEY`: Server SSH private key.
3. Update `.github/workflows/deploy.yml`:
   - Replace `your-dockerhub-username` with your actual username.
   - Update `/path/to/anemosens-api` in the deploy script to the actual path on your server.

## Daily Operations
- The application will automatically restart on push to `main` branch.
- SSL certificates will auto-renew.
