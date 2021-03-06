-- cleaning data
SELECT date (dataNotificacao) as data_notificacao,
CAST(ocupacaoCovidUti as NUMERIC) as ocupacao_CovidUti,
CAST(ocupacaoCovidCli as NUMERIC) as ocupacao_CovidCli , 
CAST(saidaConfirmadaObitos as NUMERIC) as saida_ConfirmadaObitos  , 
CAST(saidaConfirmadaAltas as NUMERIC) as saida_ConfirmadaAltas
, estado, municipio
FROM `fiap-challenge-5.covidfiap.dadoscovid`
WHERE (ocupacaoCovidUti IS NOT NULL OR  ocupacaoCovidCli IS NOT NULL) 
AND (CAST (ocupacaoCovidUti AS numeric) > 0 OR CAST (ocupacaoCovidCli AS numeric) > 0);
