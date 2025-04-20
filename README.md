# Estrutura do Projeto

```bash
.
├── config # Configurações do projeto
├── data # Dados do projeto
│   ├── bronze # Dados brutos
│   ├── gold # Dados processados
│   └── silver # Dados intermediários
├── docs # Documentação do projeto
│   └── imgs
├── log # Logs do projeto
└── src # Código fonte do projeto
    ├── notebooks # Notebooks do projeto
    │   └── eda # Exploração de Dados
    ├── packages # Pacotes do projeto
    │   ├── common # Pacote comum
    │   └── core # Pacote principal
    │       └── src
    │           └── core
    │               ├── custom_nlp # Pacote de NLP personalizado
    │               ├── data # Pacote de dados
    │               ├── models # Pacote de modelos
    │               └── visualization # Pacote de visualização
    └── tests # Testes do projeto
        ├── integration # Testes de integração
        └── unit # Testes unitários
```
