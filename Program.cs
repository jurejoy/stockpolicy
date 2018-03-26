using System;

namespace sendtest
{
    using Microsoft.Azure.EventHubs;
    using System.Text;
    using System.Threading.Tasks;

    using System;
    using Newtonsoft.Json.Linq;
    using System.Net;
    using System.Net.Http;

    class Program
    {
        private static EventHubClient eventHubClient;
        //private const string EhConnectionString = "Endpoint=sb://stockrenjie.servicebus.windows.net/;SharedAccessKeyName=stock;SharedAccessKey=4KqpkGXmE/NBv+7QehWY8lj1rh2xiJsZMbmWaLmOiy0=;EntityPath=stock";
        private const string EhConnectionString = "Endpoint=sb://stockrenjie.servicebus.chinacloudapi.cn/;SharedAccessKeyName=root;SharedAccessKey=xG9Iw/NPJGXHfo9yvmWBoAk6AVkWEWBmgo07t2pz4Mg=;EntityPath=stock";
        private const string EhEntityPath = "stock";

        static void Main(string[] args)
        {
 //           Console.WriteLine(GetStock());
            MainAsync(args).GetAwaiter().GetResult();
        }
        

        private static async Task MainAsync(string[] args)
        {
            // Creates an EventHubsConnectionStringBuilder object from the connection string, and sets the EntityPath.
            // Typically, the connection string should have the entity path in it, but this simple scenario
            // uses the connection string from the namespace.
            var connectionStringBuilder = new EventHubsConnectionStringBuilder(EhConnectionString)
            {
                EntityPath = EhEntityPath
            };




            eventHubClient = EventHubClient.CreateFromConnectionString(connectionStringBuilder.ToString());

            
            int numMessagesToSend = 10000;

            for (var i = 0; i < numMessagesToSend; i++)
            {
                string stock = GetStock();
                await SendMessagesToEventHub(stock);
            }

            await eventHubClient.CloseAsync();

            Console.WriteLine($"{numMessagesToSend} messages sent.");
            Console.WriteLine("Press ENTER to exit.");
            Console.ReadLine();
           
        }

        // Get Stock info from google finance API
        private static string GetStock()
        {
            const string tickers = "MSFT";
            string price = "";
            string json;

            using (var web = new HttpClient())
            {
                var url = $"https://finance.google.com/finance?q=NASDAQ:{tickers}&output=json";
                //               json = web.DownloadString(url);
                Task<string> response = web.GetStringAsync(url);
                json = response.Result;
            }

            //Google adds a comment before the json for some unknown reason, so we need to remove it
            json = json.Replace("//", "");

            var v = JArray.Parse(json);

            foreach (var i in v)
            {
                var ticker = i.SelectToken("t");
                var pricevar = (decimal)i.SelectToken("l");
                price = pricevar.ToString();
            }
            
            return price;
        }



        // Creates an event hub client and sends 100 messages to the event hub.
        private static async Task SendMessagesToEventHub(string stock)
        {

                try
                {
                    //var message = $"Message {i}" + " MSFT Price " + stock;
                    var message = "{'Price':" + stock + "}";
                    Console.WriteLine($"Sending message: {message}");
                    await eventHubClient.SendAsync(new EventData(Encoding.UTF8.GetBytes(message)));
                }
                catch (Exception exception)
                {
                    Console.WriteLine($"{DateTime.Now} > Exception: {exception.Message}");
                }

                await Task.Delay(10);
            

            
        }
    }
}
