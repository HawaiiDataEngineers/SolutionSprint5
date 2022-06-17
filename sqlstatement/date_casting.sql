SELECT date(date) as date, city, category_name, item_description, bottles_sold 
FROM `fiap-challenge-5.liquor_Sales.sales` 
WHERE date BETWEEN '2020-01-01' AND '2021-01-01';
