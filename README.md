# TCC

- Inicia um projeto dvc
```
dvc init
```

- Adiciona o rastreamento
```
dvc add data/second_part.csv
```

- Define o armazenamento --default (-d) para os dados
```
dvc remote add -d storing_data gdrive://1LiXbvMkEhfMvHgGBfBcybAO1XoBMycwQ
```

- Salva os dados no google drive
```
dvc pull
```