with cte_sen as
    (select sen.sentiment
            ,cast(sen.load_date as date) as [l_date]
            ,cast(sen.load_date as time) as [l_time]
            ,HOUR(sen.load_date) as [l_hour]
    from source.sentiment sen)

,cte_stonks as
    (select  stk.crypto
            ,stk.usd
            ,stk.usd_24h_change
            ,stk.usd_24h_vol
            ,cast(stk.load_date as date) as [l_date]
            ,cast(stk.load_date as time) as [l_time]
            ,HOUR(stk.load_date) as [l_hour]
    from source.stonks stk)


,cte_ins as
    (select se.sentiment
            ,st.crytpo
            ,st.usd
            ,st.usd_24h_change
            ,st.usd_24h_vol
    from cte_sen se
    inner join cte_stonks st
        on se.l_date = st.l_date
        and se.l_hour = st.l_hour
    )

insert into source.floor_table
select * from cte_ins
