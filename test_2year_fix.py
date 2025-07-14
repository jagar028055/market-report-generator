#!/usr/bin/env python3
"""Test script to verify 2-year treasury data fetching and HTML generation"""

from data_fetcher import DataFetcher
from html_generator import HTMLGenerator

def test_2year_treasury():
    print("=== Testing 2-Year Treasury Data Fetching ===")
    
    # Test data fetching
    fetcher = DataFetcher()
    market_data = fetcher.get_market_data()
    
    print(f"\nMarket data keys: {list(market_data.keys())}")
    print(f"2-year treasury data: {market_data.get('米国2年金利', 'NOT FOUND')}")
    
    # Verify the data is proper
    treasury_data = market_data.get('米国2年金利')
    if treasury_data and treasury_data['current'] != 'N/A':
        print(f"✅ 2-year treasury data is available: {treasury_data['current']}")
    else:
        print(f"❌ 2-year treasury data is missing or N/A")
        return False
    
    # Test HTML generation
    print("\n=== Testing HTML Generation ===")
    html_gen = HTMLGenerator()
    
    # Create minimal test data
    test_data = {
        'market_data': market_data,
        'sector_data': {},
        'economic_indicators': {'today': [], 'previous': []},
        'chart_data': {},
        'news_data': [],
        'commentary': {
            'stock': 'Test stock commentary',
            'bond': 'Test bond commentary', 
            'fx': 'Test FX commentary'
        }
    }
    
    try:
        html_gen.generate_report(
            market_data=market_data,
            economic_indicators={'today': [], 'previous': []},
            sector_performance={},
            news_articles=[],
            commentary='Test commentary',
            grouped_charts={}
        )
        print("✅ HTML generation completed")
        
        # Check if file was created - default filename is market_report.html
        import os
        if os.path.exists('market_report.html'):
            print("✅ HTML file created successfully")
            
            # Check if 2-year treasury data is in the HTML
            with open('market_report.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
                if '4.24' in html_content and '米国2年' in html_content:
                    print("✅ 2-year treasury data found in HTML!")
                    return True
                else:
                    print("❌ 2-year treasury data NOT found in HTML")
                    print("Looking for '米国2年債' or '米国2年金利' in HTML...")
                    if '米国2年債' in html_content:
                        print("  Found '米国2年債' in HTML")
                    if '米国2年金利' in html_content:
                        print("  Found '米国2年金利' in HTML")
                    return False
        else:
            print("❌ HTML file was not created")
            return False
            
    except Exception as e:
        print(f"❌ HTML generation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_2year_treasury()
    if success:
        print("\n🎉 All tests passed! 2-year treasury data is working correctly.")
    else:
        print("\n❌ Tests failed. Need to investigate further.")