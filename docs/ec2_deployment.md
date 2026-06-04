# AWS EC2 Production Deployment Guide

This guide details the steps required to deploy **CloudMentor AI** in a production environment on an AWS EC2 instance using Docker Compose and Nginx as a reverse proxy.

---

## 1. AWS EC2 Provisioning Requirements

### Recommended Hardware Spec
*   **Instance Type**: `t3.medium` or `t3.large` (Minimum 2 vCPUs, 4GB RAM is required for compiling vector indexes and processing PDFs).
*   **Storage**: 30 GB gp3 EBS Volume.
*   **Operating System**: Ubuntu Server 22.04 LTS (HVM).

### Security Group Rule Settings
Create or assign a Security Group exposing the following inbound ports:
*   **Port 22 (SSH)**: Restrict to your specific administrator IP addresses.
*   **Port 80 (HTTP)**: Open to `0.0.0.0/0` (public web access).
*   **Port 443 (HTTPS)**: Open to `0.0.0.0/0` (secure public web access).

---

## 2. Server Installation & Configuration

Connect to your EC2 instance via SSH and run the installation script:

```bash
# Update system dependencies
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Ensure current user can run Docker commands without sudo
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. Cloning Codebase & Configuring Secrets

```bash
# Clone the repository
git clone <your-git-repository-url> cloudmentor-ai
cd cloudmentor-ai

# Set up local directories
mkdir -p data/chromadb data/embeddings_cache docs

# Configure production environment variables
cp .env.example .env
nano .env
# Edit GEMINI_API_KEY with your production Google Gemini key
```

---

## 4. Run the Application Services

Use Docker Compose to build and execute the application container:

```bash
# Build the image and start the application container in detached mode
docker compose up -d --build

# Verify container status
docker compose ps

# Check execution logs
docker compose logs -f
```

The application is now running locally on port `8501`.

---

## 5. Exposing Secure Traffic (Nginx Reverse Proxy & SSL)

For production, we proxy traffic from public port 80/443 to Docker container port 8501, securing it with TLS certificates.

### Install Nginx & Certbot
```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

### Configure Nginx Configuration
Create a configuration file: `/etc/nginx/sites-available/cloudmentor`
```nginx
server {
    listen 80;
    server_name cloudmentor.yourdomain.com; # Replace with your subdomain

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/cloudmentor /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### Register SSL Certificates
Execute Certbot to retrieve Let's Encrypt certificates and configure HTTPS automatically:
```bash
sudo certbot --nginx -d cloudmentor.yourdomain.com
```

Select option `2` to redirect all HTTP traffic to HTTPS.

---

## 6. Backup and Maintenance

*   **Database persistence**: ChromaDB persistent data is saved to `/app/data/chromadb` inside the container, which is volume-mapped to the host's `./data/chromadb` directory. Do not delete this directory to prevent document index losses.
*   **Backups**: Set up a cron task to perform daily archives of the `./data` directory to Amazon S3:
    ```bash
    tar -czf db_backup_$(date +%F).tar.gz ./data/chromadb
    aws s3 cp db_backup_*.tar.gz s3://your-backup-bucket/
    ```
