#!/bin/bash

# 🚀 Safe Push Script - Dataset Protection Enabled
# This script safely pushes code while protecting dataset files

echo "🔍 PRE-PUSH SAFETY CHECKS"
echo "=========================="

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository"
    exit 1
fi

# Check for dataset files in staging area
echo "📋 Checking for dataset files in staging area..."
DATASET_FILES=$(git diff --cached --name-only | grep -E "(dataset|\.dat|\.csv|\.inter|\.pkl|\.log|yoochoose|diginetica)" || true)

if [ ! -z "$DATASET_FILES" ]; then
    echo "🚨 WARNING: Dataset files detected in staging area:"
    echo "$DATASET_FILES"
    echo ""
    echo "❌ PUSH BLOCKED - Dataset files must not be pushed to GitHub!"
    echo "💡 Please unstage these files: git reset HEAD <file>"
    exit 1
fi

# Check for large files (>10MB)
echo "📏 Checking for large files..."
LARGE_FILES=$(git diff --cached --name-only | xargs -I {} sh -c 'if [ -f "{}" ]; then find "{}" -size +10M; fi' 2>/dev/null || true)

if [ ! -z "$LARGE_FILES" ]; then
    echo "⚠️  WARNING: Large files detected (>10MB):"
    echo "$LARGE_FILES"
    read -p "🤔 Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Push cancelled"
        exit 1
    fi
fi

# Show current status
echo "📊 Current Git Status:"
echo "----------------------"
git status --short

echo ""
echo "📝 Files to be committed:"
echo "-------------------------"
git diff --cached --name-status

# Commit with auto-generated message or user input
echo ""
read -p "💬 Enter commit message (or press Enter for auto-message): " COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    COMMIT_MSG="🔄 Update: $TIMESTAMP"
fi

# Add files (excluding datasets - .gitignore will handle this)
echo ""
echo "📦 Adding files to staging..."
git add .

# Final check after adding
echo "🔍 Final dataset check after staging..."
FINAL_CHECK=$(git diff --cached --name-only | grep -E "(dataset|\.dat|\.csv|\.inter|\.pkl|yoochoose|diginetica)" || true)

if [ ! -z "$FINAL_CHECK" ]; then
    echo "🚨 CRITICAL: Dataset files still in staging after .gitignore!"
    echo "$FINAL_CHECK"
    echo "❌ PUSH BLOCKED - Please check .gitignore configuration"
    exit 1
fi

# Commit changes
echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo "❌ Commit failed"
    exit 1
fi

# Push to origin
echo "🚀 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: Code pushed safely to GitHub!"
    echo "🛡️  Dataset protection: ACTIVE"
    echo "📈 Repository updated successfully"
else
    echo ""
    echo "❌ Push failed - please check connection and permissions"
    exit 1
fi