from flask import Flask, request, render_template, session, url_for, redirect
from flask_session import Session
import socket
import backtrader 
import datetime
import random
import os


graphdestination = os.path.join('static')

def saveplots(cerebro, numfigs=1, iplot=True, start=None, end=None,
        width = 16, height = 9, dpi=300, tight=True, use=None, file_path = '', **kwargs):

        from backtrader import plot
        if cerebro.p.oldsync:
            plotter = plot.Plot_OldSync(**kwargs)
        else:
            plotter = plot.Plot(**kwargs)

        figs = []
        
        for stratlist in cerebro.runstrats:
            
            for si, strat in enumerate(stratlist):
                
                rfig = plotter.plot(strat, figid=si * 100,
                                    numfigs=numfigs, iplot=iplot,
                                    start=start, end=end, use=use)
                figs.append(rfig)

        
        for fig in figs:
            for f in fig:
                f.savefig(file_path, bbox_inches='tight')
        return figs

app = Flask(__name__)
app.debug = True
# list of all possible stock tickers accepted by app
stock_tickers = ["AAPL",
        "MSFT",
        "AMZN",
        "GOOG",
        "IBM",
        "SNE", 
        "FB",
        "TSLA",
        "NFLX",
        "ADBE",
        "QCOM",
        "SAP",
        "DBX",
        "NVDA",
        "AMD",
        "XRX",
        "IT"]

@app.route('/', methods = ['GET', 'POST'])
def index():
    
    if request.method == 'POST': # user has submitted a stock ticker
        # use globals for use between functions
        global stock_name
        stock_name = request.form.get("searchbar")
        
        if stock_name in stock_tickers: # check validity 
            
            import strategy
            strategy.stock_name = stock_name
            saveplots(strategy.cerebro, file_path = os.path.join(graphdestination, 'pnl.png'))
            starting_value = strategy.starting_portfolio_value
            ending_value = strategy.final_portfolio_value
            return redirect(url_for('graph',
                    starting_value = int(starting_value),
                    ending_value = int(ending_value)
                    )) # generate plot and description
            
        
        else:
            return render_template(
                "index.html",
                error_message = "No time-series data exists for this ticker. Please try again with a valid stock ticker like " + str(random.choice(stock_tickers)))

    return render_template("index.html")


@app.route('/graph/<starting_value>/<ending_value>', methods = ['GET', 'POST'])

def graph(starting_value, ending_value):
    
    starting_value = int(starting_value)
    ending_value = int(ending_value)
    returns = ((ending_value - starting_value) / starting_value) * 100
    yearly_returns = returns / 4
    return render_template(
            "graph.html",
            pnl_chart = url_for('static', filename = 'pnl.png'),
            returns = str(returns),
            starting_value = str(starting_value),
            ending_value = str(ending_value),
            yearly_returns = yearly_returns) 
