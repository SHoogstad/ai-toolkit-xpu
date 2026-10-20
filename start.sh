#!/bin/bash
set -e
cd /app/ai-toolkit/ui
# create/upgrade the sqlite schema in the bind-mounted DB on every start
npx prisma db push --skip-generate
exec npm run start
