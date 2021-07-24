insert into source.base_data with cte as (SELECT s.crypto,
                                                 s.id,
                                                 s.usd,
                                                 s.usd_24h_change,
                                                 s.usd_24h_vol,
                                                 s.usd_market_cap,
                                                 s2.load_date,
                                                 avg(s2.score),
                                                 row_number() OVER (PARTITION BY s.id
                                                                    ORDER BY s.id) row_num
                                          FROM   stonks s
                                              INNER JOIN sentiment s2
                                                  ON date_format(s.load_date, '%Y-%m-%d %H:%M:00') = date_format(s2.load_date, '%Y-%m-%d %H:%M:00')
                                          WHERE  s.crypto = 'bitcoin'
                                             and s2.load_date between date_format(now(), '%Y-%m-%d %H:%M:00') - interval 30 minute
                                             and date_format(now(), '%Y-%m-%d %H:%M:00')
                                          GROUP BY s.crypto, s.id, s.usd, s.usd_24h_change, s.usd_24h_vol, s.usd_market_cap, s2.load_date), scores as (SELECT date_format(load_date, '%Y-%m-%d %H:%M:00') load_dt,
                                                                                                                    avg(cast(left(label, 1) as float)) avg_rating
                                                                                                             FROM   sentiment s
                                                                                                             WHERE  score > 0.7
                                                                                                             GROUP BY load_dt)
SELECT cte.*,
       scores.avg_rating
FROM   cte
    INNER JOIN scores
        ON date_format(cte.load_date, '%Y-%m-%d %H:%M:00') = date_format(scores.load_dt, '%Y-%m-%d %H:%M:00')
WHERE  row_num = 1 limit 1;
