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
