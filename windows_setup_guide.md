# Elytics - Complete Windows Setup Guide

Follow these steps **exactly in order** after extracting the project on your Windows laptop.

---

### Step 1: Open Terminal
1. Open the extracted `Elytics` folder.
2. Right-click inside the folder and select **Open in Terminal** (this opens PowerShell).

---

### Step 2: Setup the Python Backend
In your PowerShell window, run these commands one by one:

1. **Go into the backend folder:**
   ```powershell
   cd backend
   ```

2. **Create a new Virtual Environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   *(You must do this every time before running the backend)*
   ```powershell
   .\venv\Scripts\activate
   ```
   *(You should see `(venv)` appear on the left side of your terminal prompt)*

4. **Upgrade PIP (recommended by your Windows machine):**
   ```powershell
   python -m pip install --upgrade pip
   ```

5. **Install all required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

---

### Step 3: Setup your Credentials
1. Inside the `backend` folder, find the file named `.env.example`.
2. Rename it to exactly `.env` (with the dot at the beginning).
3. Open it in a text editor and fill in your AWS / Redshift credentials.

---

### Step 4: Start the Backend Server
Make sure you are still in the `backend` folder and `(venv)` is still active, then run:
```powershell
python -m uvicorn app.main:app --reload
```
Leave this terminal window open and running.

---

### Step 5: Setup the React Frontend
1. Keep the first terminal running! Open a **BRAND NEW** PowerShell window.
2. Navigate to the `Elytics\frontend` folder:
   ```powershell
   cd path\to\Elytics\frontend
   ```

3. **Install the Node dependencies:**
   *(This downloads everything into a local node_modules folder)*
   ```powershell
   npm install
   ```

---

### Step 6: Start the Frontend
In that same frontend terminal, run:
```powershell
npm run dev
```

### Step 7: Open the App
Open your web browser (Chrome/Edge) and go to:
**http://localhost:5173**

You are all set!
