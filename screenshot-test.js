const { chromium } = require('playwright');
const path = require('path');

async function takeScreenshot() {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    
    // タイムスタンプを生成
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const screenshotsDir = 'screenshots';
    
    try {
        // HTMLファイルのファイルパス
        const filePath = 'file://' + path.resolve('./index.html');
        console.log('Loading page:', filePath);
        
        // ページを読み込み
        await page.goto(filePath, { waitUntil: 'networkidle' });
        
        // ページがロードされるまで少し待つ
        await page.waitForTimeout(2000);
        
        console.log('Page loaded successfully');
        
        // 全ページのスクリーンショット
        const fullPagePath = `${screenshotsDir}/${timestamp}_fullpage.png`;
        await page.screenshot({ 
            path: fullPagePath, 
            fullPage: true 
        });
        console.log(`Full page screenshot saved as ${fullPagePath}`);
        
        // S&P500カードをクリック
        console.log('Looking for S&P500 card...');
        const sp500Card = await page.locator('[data-indicator="S&P500"]').first();
        
        if (await sp500Card.isVisible()) {
            console.log('S&P500 card found, clicking...');
            await sp500Card.click();
            
            // モーダルが開くまで待つ
            await page.waitForSelector('.market-modal.show', { timeout: 5000 });
            console.log('Modal opened successfully');
            
            // モーダルのスクリーンショット
            const modalPath = `${screenshotsDir}/${timestamp}_modal-opened.png`;
            await page.screenshot({ 
                path: modalPath 
            });
            console.log(`Modal screenshot saved as ${modalPath}`);
            
            // チャートが読み込まれるまでより長く待つ
            await page.waitForTimeout(5000);
            
            // チャート画像が完全に読み込まれるまで待つ
            await page.waitForFunction(() => {
                const chartImg = document.querySelector('#modalChartFrame');
                return chartImg && chartImg.complete && chartImg.naturalHeight > 0;
            }, { timeout: 10000 });
            
            // チャート読み込み後のスクリーンショット
            const chartPath = `${screenshotsDir}/${timestamp}_modal-with-chart.png`;
            await page.screenshot({ 
                path: chartPath 
            });
            console.log(`Modal with complete chart screenshot saved as ${chartPath}`);
        } else {
            console.log('S&P500 card not found');
        }
        
    } catch (error) {
        console.error('Error during screenshot process:', error);
    }
    
    await browser.close();
}

takeScreenshot();