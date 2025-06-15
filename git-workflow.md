# Git Branching Strategy

---

## 📚 Branches

- **`main`** – Stable, production-ready code only.
- **`develop`** – Active development happens here.

---

## 🚀 Workflow

### 1. Create `develop` from `main`
```bash
git checkout main
git pull
git checkout -b develop
```

### 2. Work on develop
Commit your changes, build features, and test everything on the develop branch.

### 3. Rebase develop onto main before merging
```bash
git checkout develop
git fetch origin
git rebase origin/main
```

### 4. Merge to main (fast-forward only)
```bash
git checkout main
git merge --ff-only develop
git push origin main
```

## 🛠 Optional Enhancements
Create feature branches from develop:

```bash
git checkout develop
git checkout -b feature/my-feature
```

Use Pull Requests (PRs) if you're working in a team.

Protect main with required reviews or CI checks.

## ✅ Benefits
- Keeps main always deployable
- Clean and linear Git history with rebasing
- Easy to manage and understand

## 📌 Notes
- Always run tests before merging to main.
- Use --ff-only to avoid unnecessary merge commits.