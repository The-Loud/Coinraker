SELECT *
FROM   (SELECT usd,
               usd_24h_change,
               usd_24h_vol,
               usd_market_cap,
               load_date,
               avg_rating
        FROM   source.base_data
        ORDER BY load_date desc limit 24) sub
ORDER BY load_date asc;
