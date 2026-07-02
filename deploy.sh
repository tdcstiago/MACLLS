#!/bin/bash
git pull origin main
docker-compose up -d --build --force-recreate
sudo nginx -t && sudo systemctl reload nginx
