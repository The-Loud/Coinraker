insert into source.base_data with cte as (SELECT s.crypto ,
                                                 s.id ,
                                                 s.usd ,
                                                 s.usd_24h_change ,
                                                 s.usd_24h_vol ,
                                                 s.usd_market_cap ,
                                                 s2.load_date ,
                                                 avg(s2.score) ,
                                                 row_number() OVER (PARTITION BY s.id
                                                                    ORDER BY s.id) row_num
                                          FROM   stonks s
                                              INNER JOIN sentiment s2
                                                  ON date_format(s.load_date, '%Y-%m-%d %H:00:00') = date_format(s2.load_date, '%Y-%m-%d %H:00:00')
                                          WHERE  s.crypto = 'bitcoin'
                                             and s2.load_date between date_format(now(), '%Y-%m-%d %H:00:00') - interval 1 hour
                                             and date_format(now(), '%Y-%m-%d %H:00:00')
                                          GROUP BY s.crypto , s.id , s.usd , s.usd_24h_change , s.usd_24h_vol , s.usd_market_cap , s2.load_date)
SELECT *
FROM   cte
WHERE  row_num = 1;
