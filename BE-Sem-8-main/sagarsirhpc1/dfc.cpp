#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;
const int MAXN = 1e5;
vector<int> adj[MAXN + 5]; // adjacency list
bool visited[MAXN + 5];    // mark visited nodes
void dfs(int node)
{
    visited[node] = true;
#pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++)
    {
        int next_node = adj[node][i];
        if (!visited[next_node])
        {
            dfs(next_node);
        }
    }
}
int main()
{
    cout << "Please enter nodes and edges";
    int n, m; // number of nodes and edges
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        int u, v; // edge between u and v
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int start_node; // start node of DFS
    cin >> start_node;
    dfs(start_node);
    // Print visited nodes
    for (int i = 1; i <= n; i++)
    {
        if (visited[i])
        {
            cout << i << " ";
        }
    }
    cout << endl;
    return 0;
}