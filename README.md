# MiniAI 1.0 Standalone

This project packages your supplied `MiniAI_1.0.onnx` model into:
- a static web application,
- an installable PWA with an offline app shell,
- an Android APK build workflow using Capacitor and GitHub Actions.

## Important model details
The model is the 8x8 sklearn Digits classifier: 64 float inputs, 10 output logits.
The original training normalized the 0..16 grayscale pixels by dividing by 16.
The app converts its 8-bit grayscale canvas to 0..1 values, matching that scale.

## Deploy the website with GitHub Pages
1. Create a GitHub repository, e.g. `MiniAI-Standalone`.
2. Upload the contents of this ZIP to the repository root (keep `.github/workflows`).
3. Set the default branch to `main`.
4. Open Settings → Pages and select **GitHub Actions** as the build/deployment source.
5. Open Actions and wait for **Deploy MiniAI Web** to complete.
6. Open the Pages URL. The first visit needs internet to download runtime assets. The service worker caches the app shell and model for later offline use.

## Build the Android APK
1. Push the project to GitHub.
2. Open Actions → **Build MiniAI Android APK** → Run workflow.
3. Wait for the workflow to finish.
4. Open its run → Artifacts → download `MiniAI-Android-debug`.
5. Extract the ZIP and install `app-debug.apk` on your Android device. Allow installation from your browser/file manager if Android asks.
6. This is a debug APK for personal testing, not a Play Store release. A signed release requires your own signing key.

## Test offline
For the PWA, open the hosted site once while online and wait until the AI status says **AI model ready**. Then reload once, and test with airplane mode. The service worker caches the app files and model. The Android APK bundles the web assets and model locally.

## Notes
- The browser uses ONNX Runtime Web WASM.
- This is local inference; drawings are not sent to a server.
- The Google Fonts import is decorative; if offline, the app falls back to system fonts.
- The digit recognizer was trained on 8x8 grayscale images. Handwriting style and preprocessing can affect accuracy.
- Do not rename or remove `MiniAI_1.0.onnx`.
