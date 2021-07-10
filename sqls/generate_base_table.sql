with cte_sen as (SELECT sen.sentiment ,
                        cast(sen.load_date as date) [l_date] ,
                        cast(sen.load_date as time) [l_time] ,
                        hour(sen.load_date) [l_hour]
                 FROM   source.sentiment sen) , cte_stonks as (SELECT stk.crypto ,
                                                     stk.usd ,
                                                     stk.usd_24h_change ,
                                                     stk.usd_24h_vol ,
                                                     cast(stk.load_date as date) as [l_date] ,
                                                     cast(stk.load_date as time) as [l_time] ,
                                                     hour(stk.load_date) as [l_hour]
                                              FROM   source.stonks stk) , cte_ins as (SELECT se.sentiment ,
                                               st.crytpo ,
                                               st.usd ,
                                               st.usd_24h_change ,
                                               st.usd_24h_vol
                                        FROM   cte_sen se
                                            INNER JOIN cte_stonks st
                                                ON se.l_date = st.l_date and
                                                   se.l_hour = st.l_hour);


insert into source.floor_table
SELECT *
FROM   cte_ins;
