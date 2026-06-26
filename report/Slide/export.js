const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

(async () => {

    const browser = await chromium.launch({
        headless: true
    });

    const page = await browser.newPage({
        viewport: {
            width: 1440,
            height: 1080
        },
        deviceScaleFactor: 3      // Xuất 4320×3240
    });

    await page.goto(
        "file://" + path.resolve("MLSlide.html"),
        {
            waitUntil: "networkidle"
        }
    );

    // Đợi font
    await page.evaluate(async () => {
        await document.fonts.ready;
    });

    // Đợi toàn bộ ảnh
    await page.evaluate(async () => {

        const imgs = [...document.images];

        await Promise.all(imgs.map(img => {

            if (img.complete) return;

            return new Promise(resolve => {
                img.onload = resolve;
                img.onerror = resolve;
            });

        }));

    });

    // Đợi MathJax render

    await page.waitForTimeout(3000);

    if (await page.evaluate(() => window.MathJax !== undefined)) {

        await page.evaluate(async () => {

            await MathJax.typesetPromise();

        });

    }

    fs.mkdirSync("output", {
        recursive: true
    });

    const slides = await page.locator(".slide").count();

    console.log(`Found ${slides} slides`);

    for (let i = 0; i < slides; i++) {

        console.log(`Rendering slide ${i + 1}`);

        await page.evaluate((index) => {

            const slides = [...document.querySelectorAll(".slide")];

            slides.forEach(s => {

                s.classList.remove("active");

                s.style.display = "none";

            });

            slides[index].classList.add("active");

            slides[index].style.display = "block";

        }, i);

        await page.waitForTimeout(300);

        await page.locator(".slide.active").screenshot({

            path: `output/slide-${String(i + 1).padStart(2, "0")}.png`

        });

    }

    await browser.close();

    console.log("Done!");

})();