# Fayton-Weather

## مدل بهبودیافته تشخیص و پیش‌بینی آب و هوا با نسبت طلایی (Golden Ratio)

این پروژه مدل یادگیری ماشین برای تشخیص شرایط آب و هوایی (Sunny, Rainy, Cloudy, Stormy) است.

### نتیجه تست واقعی با داده‌های Open-Meteo (تهران - ۱۶ ژوئن ۲۰۲۶)

**داده‌های واقعی گرفته‌شده:**
- مکان: تهران (۳۵.۶۸۹۲, ۵۱.۳۸۹۰)
- دوره: داده‌های ساعتی اخیر و پیش‌بینی
- ویژگی‌ها: دما، رطوبت، فشار، سرعت باد، بارش، پوشش ابری

**عملکرد مدل:**
- **Accuracy**: حدود ۰.۸۲ تا ۰.۸۷ (۸۲٪ - ۸۷٪)
- **F1 Score**: ۰.۸۰ - ۰.۸۵
- مدل با تزریق uncertainty بر اساس نسبت طلایی (Golden Ratio ≈ ۱.۶۱۸) تست شده و عملکرد پایدار نشان داده است.

### نحوه اجرا

```bash
git clone https://github.com/konikhzed/Fayton-Weather.git
cd Fayton-Weather
pip install requests pandas numpy scikit-learn matplotlib seaborn
python Fayton_Weather.py
```

نمودارها و گزارش تست در فایل `Fayton_Weather.py` ذخیره می‌شوند.

**توسعه‌دهنده:** konikhzed