# Gene_Corpus_Dashboard

## Code and local testing

The main file is [./gene_corpus_dashboard.qmd](./gene_corpus_dashboard.qmd).

You can run the following to preview the dashboard in localhost:
```
quarto preview neurotrial_dashboard.qmd
```

The command should also result in the following outputs:

```
neurotrial_dashboard.html
neurotrial_dashboard_files/
app.py
```

You can run this as a normal Shiny application with shiny run to make sure it works as expected.
```
shiny run
```

## Deployment to shinyapps.io

As described in [Cloud hosting](https://shiny.posit.co/py/docs/deploy-cloud.html):

```
rsconnect deploy shiny /path/to/app --name <NAME> --title my-app
```

where:
- < NAME >: This specifies the account or server name that you have set up in your rsconnect configuration. It determines where the app will be deployed.
- my-app: This sets the title of the application as it will appear in the deployment environment.
