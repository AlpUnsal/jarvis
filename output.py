import webbrowser

def open_url(url):
    """Opens a URL in the default web browser."""
    webbrowser.open(url)

def main():
    """Opens Gmail in the default web browser."""
    url = "https://www.gmail.com"
    open_url(url)

if __name__ == "__main__":
    main()