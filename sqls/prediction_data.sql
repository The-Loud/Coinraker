SELECT usd,
       usd_24h_change,
       usd_24h_vol,
       usd_market_cap,
       load_date,
       avg_rating
FROM   source.base_data
WHERE  load_date between date_format(now(), '%%Y-%%m-%%d %%H:00:00') - interval 24 hour
   and date_format(now(), '%%Y-%%m-%%d %%H:00:00');
