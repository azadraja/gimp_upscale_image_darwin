# GIMP AI Upscaler for macOS 

A powerful GIMP plugin that integrates the Real-ESRGAN AI model to upscale your images with incredible detail. This version has been specifically packaged to run locally on **macOS**.

This is a fork of the excellent [gimp_upscale plugin by Nenotriple](https://github.com/Nenotriple/gimp_upscale), modified to work with a native macOS executable.

---

## ⚠️ Requirements

Before installing, please make sure you have the following:

* **Operating System:** **macOS** (This plugin is not compatible with Windows or Linux).
* **GIMP Version:** **GIMP 3.0** or newer. This plugin will *not* work with GIMP 2.10.

---

## ✨ Features

* **Local Processing:** All upscaling is done on your machine. No internet connection is needed and your images remain private.
* **Choose Your Model:** Select from 6 powerful, built-in AI models to get the best result for your specific image:
    * `realesr-animevideov3-x4` (Great for anime and videos)
    * `RealESRGAN_General_x4_v3`
    * `realesrgan-x4plus` (A powerful all-rounder)
    * `realesrgan-x4plus-anime` (Excellent for illustrations and anime art)
    * `UltraSharp-4x`
    * `AnimeSharp-4x`
* **Bring Your Own Models:** 🧪 Experiment by adding your own compatible `.bin` and `.param` model files to the `resrgan/models` folder.
* **Flexible Output:** Control the final canvas size with an **Output Factor** ranging from 0.1x to 8x.
* **Multiple Scopes:** Apply the upscale to the entire image, just the current layer, or only the area within a selection.

---

## 🚀 Installation

Follow these steps carefully to install the plugin.

### 1. Clone the Repository

Open your **Terminal** app and run the following command to download the plugin files:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
```
### 2. Copy to GIMP's Plug-ins Folder

Copy the entire plugin folder into GIMP's `plug-ins` directory.

* Open **Finder**.
* In the menu bar at the top, click **Go**, hold down the **Option (⌥)** key, and click on **Library**.
* Navigate to: `Application Support/GIMP/3.0/plug-ins/`
* Drag and drop the entire cloned folder into this `plug-ins` directory.

### 3. Set Permissions

Navigate into the plugin's `resrgan` folder in your Terminal and make the AI engine executable. This step is crucial!

```bash
# Navigate to the correct directory (adjust the path if needed)
cd ~/Library/Application\ Support/GIMP/3.0/plug-ins/your-repo-name/resrgan/

# Make the engine executable
chmod +x gimp3_upscale.py
```

### 4. Restart GIMP

Completely quit and restart GIMP. It only scans for new plugins when it starts up.

---

## 🎨 How to Use

1.  Open any image in GIMP.
2.  Go to the main menu bar: **Filters > Enhance > AI Upscale...**
3.  A dialog box will appear.
    * Choose the AI model you want to use.
    * Select the scope (Entire image, Selection only, or Layer only).
    * Adjust the Output Factor if needed.
4.  Click **OK** and wait for the magic to happen! A new, upscaled layer will be added to your project.

---

## 🙏 Acknowledgements

* **Original Plugin:** A huge thank you to [Nenotriple](https://github.com/Nenotriple/gimp_upscale) for creating the original GIMP plugin.
* For Windows and Linux versions, please see [Nenotriple's original repository](https://github.com/Nenotriple/gimp_upscale).
* **AI Engine:** This plugin uses the powerful **Real-ESRGAN-ncnn-vulkan** executable created by [xinntao](https://github.com/Nenotriple](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)).

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

