#include <bits/stdc++.h>
#include <chrono>
using namespace std;
auto strt = chrono::steady_clock::now();

#define ll long long
const int mod = 1e9 + 7;
#define __power(x, y) pow(x, y);
#define __sqroot(x) sqrt(x);
#define __sqrn(x) pow(x, 2);
#define __gcd(x, y) __gcd(x, y);
#define __lcm(x, y) (x * y) / __gcd(x, y);
#define __ceil(x) ceil(x);   // 5.678 -- 6
#define __floor(x) floor(x); // 5.678--5
#define __max(x, y) max(x, y);
#define __min(x, y) min(x, y);
#define __mxmi(x, y) (x > y) ? x : y;
#define __abs(x) abs(x);                       // NEG[-]->POS[+]
#define __CountBit_n(n) __builtin_popcount(n); // count no of 1's in n
#define __Get_sen(s) getline(cin, s);          // string si; __Get_sen(si);
#define substr(x, y) substr(x, y);             // msg.substr(3, 4)  // indx 3 to size of 4 string
#define insert(x, y) insert(x, y);             // msg.insert(3, "insert")  // insert after pos 3

#define inVC_1n(i, st, en) for (int i = st; i < en; i++)
#define inVC_n1(i, rst, ren) for (int i = rst - 1; i >= ren; i--)
#define all(x) x.begin(), x.end() //  sort(all(vc));
/*
 vector<int>vc;
  inVC_1n(i,3,9){
  cout<<i<<" ";
  vc.push_back(i);
  }
*/
#define unique(arr, n) unique(arr, arr + n)
#define pq_pair_sort(x, y) priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

#define yes() cout << "YES" << endl
#define no() cout << "NO" << endl

/*....................STL...................*/
// #define binary_search(vc.begin(),vc.end(),66)

bool myComp(const pair<ll, ll> &a, const pair<ll, ll> &b)
{
  if (a.first != b.first)
    return a.first > b.first;
  return a.second < b.second;
}

int sizArr(int a)
{
  return sizeof(a) / sizeof(int);
};

#define substr(x, y) substr(x, y); // msg.substr(3, 4)  // indx 3 to size of 4 string
#define insert(x, y) insert(x, y); // msg.insert(3, "insert")  // insert after pos 3

//(binary_search(arr.begin()+i, arr.end(), k))

/*******************************************PROSESS*************************************************/

/*GOTO
void fun(int m)
{
    cout << m + 5;
}
int main()
{
    int n;
    cin >> n;
    int m = n;
    for (int i = 0; i < n; i++)
    {
        cout << i << " ";
        if (i % 2 == 0)
        {
            goto r2;
            goto r1;
            continue;
        }
        cout << i << " ";
    }
r2:
    fun(m);
r1:
    cout << "love";
    return 0;
}
*/

/***************************************MATH******************************************************/
string to_binary(int n)
{
  int decimal = n;
  const int k = 10;

  string binary = bitset<k>(decimal).to_string(); // bitset<k>(n).to_string()
  return binary;                                  /*Binary string to INT   stoi(binary)*/
}

/*Parity: Parity of a number refers to whether it contains an odd or even number of 1-bits.
 The number has “odd parity” if it contains an odd number of 1-bits*/
bool getParity(unsigned int n)
{
  // return __builtin_parity(n);
  bool parity = 0;
  while (n)
  {
    parity = !parity;
    n = n & (n - 1);
  }
  return parity;
}

const int N = 5e5;
bool seive[N];

int Combination_ll2pow(int n)
{
  ll k = 1;
  for (int i = 1; i <= n; i++)
  {
    k = (k * 2) % mod;
  }
  return k;
}

void seivePRIM()
{
  for (int i = 2; i < N; ++i)
  {
    seive[i] = true;
  }

  for (int i = 2; i < N; ++i)
  {
    if (seive[i])
    {
      for (long long j = i * 1LL * i; j < N; j = j + i)
      {
        seive[j] = false;
      }
    }
  }
}

bool isPrime(int n)
{
  // since 0 and 1 is not prime return false.
  if (n == 1 || n == 0)
    return false;
  // Run a loop from 2 to square root of n.
  for (int i = 2; i * i <= n; i++)
  {
    // if the number is divisible by i, then n is not a prime number.
    if (n % i == 0)
      return false;
  }
  // otherwise, n is prime number.
  return true;
}
void primeNo(int n)
{
  for (int i = 1; i <= n; i++)
  {
    // check if current number is prime
    if (isPrime(i))
    {
      cout << i << " ";
    }
  }
}
bool is_fib(int n)
{
  if (n <= 2 && n >= 0)
  {
    return 1;
  }
  int a = 0, b = 1, c;
  while (1)
  {
    c = a + b;
    a = b;
    b = c;
    if (c == n)
    {
      return 1;
    }
    else if (c > n)
    {
      return 0;
    }
  }
}

// ! what will be the mathamatical Sign ( + , - )
int sign(int up, int lw)
{
  int sign = ((up < 0) ^ (lw < 0)) ? -1 : 1;

  /*

  up/lw
  >> - * - = +
  >> + * - = -
  >> - * + = -
  >> + * + = +

  */
}

bool ispow2(int n)
{
  if (n <= 0)
  {
    return 0;
  }
  return ((n & (n - 1)) == 0);
  /*n=8 = 1000
    n=n-1 = 7 = 0111
  (  n&(n-1)==0 )  power of two T.C (1) S.C (1)
    */
}

/*trvels the K x K square matrix where give the index of this square matrix left upper corner [row][col]
int row=4,col=5;
 for(int i=0; i<68; i++){
  cout<<(k * (row / k) + (i / k))<<"  " <<(k * (col / k) + (i % k))<<endl;
 }

*/

/*MATRIX FORMAT 4 DIRECTION :: */
int dx4[4] = {1, -1, 0, 0};
int dy4[4] = {0, 0, 1, -1};

/*MATRIX 8 DIRECTION :: */
int dx8[8] = {1, 1, 1, -1, -1, -1, 0, 0};
int dy8[8] = {1, 0, -1, 1, -1, 0, 1, 1};

bool valid_D(int x, int y, int n, int m)
{
  return (x >= 0 && x < n && y >= 0 && y < m);
}

/*Hourse step---*/
int dx[] = {-2, -1, 1, 2, -2, -1, 1, 2};
int dy[] = {-1, -2, -2, -1, 1, 2, 2, 1};
bool valid_H(int x, int y, int N)
{
  if (x >= 1 && x <= N && y >= 1 && y <= N)
    return true;
  return false;
}

vector<vector<int>> sort_matrix_by_last_valueof_Row(const vector<vector<int>> &matrix)
{

  vector<vector<int>> sorted_matrix = matrix;
  sort(sorted_matrix.begin(), sorted_matrix.end(), [](const auto &a, const auto &b)
       { return a[1] > b[1]; });

  return sorted_matrix;
}

vector<vector<int>> all_prmut;
/*number of ways it can be ordered or arranged*/
void all_permut(vector<int> a)
{
  sort(a.begin(), a.end());
  do
  {
    all_prmut.push_back(a);
  } while (next_permutation(a.begin(), a.end()));
}

vector<vector<int>> all_sbst;
/*any possible combination of the original array (or a set)*/
void all_subset(vector<int> &nums, int n, int idx, vector<int> &tem)
{
  all_sbst.push_back(tem);
  for (int i = idx; i < n; i++)
  {
    tem.push_back(nums[idx]);
    all_subset(nums, n, i + 1, tem);
    tem.pop_back();
  }
}

vector<vector<int>> all_sbar;
/*a contiguous part of array*/
void all_subarr(vector<int> &nums)
{
  int n = nums.size();
  for (int i = 0; i < n; i++)
  {
    for (int j = i; j < n; j++)
    {
      vector<int> in;
      for (int k = i; k <= j; k++)
      {
        in.push_back(nums[k]);
      }
      all_sbar.push_back(in);
    }
  }
}

vector<vector<int>> all_subsq;
/*Elements having the same sequential ordering*/
void all_subseq(vector<int> &vc)
{
  int n = vc.size();
  /* Number of subsequences is (2**n -1)*/
  unsigned int opsize = pow(2, n);
  /* Run from counter 000..1 to 111..1*/
  for (int counter = 1; counter < opsize; counter++)
  {
    vector<int> in;
    for (int j = 0; j < n; j++)
    {
      /* Check if jth bit in the counter is set
            If set then print jth element from arr[] */
      if (counter & (1 << j))
        in.push_back(vc[j]);
    }
    all_subseq(in);
  }
}

bool find_ch_String(string str, char ch)
{
  if (str.find(ch) != string::npos)
  {
    return 1;
    // cout << "Character " << ch << " is present in the string." << endl;
  }
  return 0;
}

bool can_form_palindrome(int arr[], int n)
{
  string str = "";
  for (int i = 0; i < n; i++)
  {
    str += arr[i];
  }
  map<int, int> freq;
  for (int i = 0; str[i]; i++)
  {
    freq[str[i]]++;
  }
  int count = 0;

  for (int i = 0; i < freq.size(); i++)
  {
    if (freq[i] & 1)
    {
      count++; // Count odd occurring characters
    }
    if (count > 1)
    {
      return false;
    }
  }
  // Return true if odd count is 0 or 1,
  return true;
}

/* string merge_rmv_overlap(const string& a, const string& b) {
        if (a.find(b) != string::npos)
        {
            return a;
        }

        for (int i = min(a.length(), b.length()); i >= 0; i--) {
            if (a.substr(a.length() - i) == b.substr(0, i)) {
                return a + b.substr(i);
            }
        }

        return a + b;
    }*/

/*
msg.substr(3,4)
msg.append("End")



Comparing with another string:
int compare(string another); Compare the content of this string with
the given another & return 0 if equals;
a negative value if this string is less than another; positive value otherwise.

== and != Operators   Compare the contents of two strings

string str1("Hello"), str2("Hallo"), str3("hello"), str4("Hello");
cout << str1.compare(str2) << endl;   // 1   'e' > 'a'
cout << str1.compare(str3) << endl;   // -1  'h' < 'H'
cout << str1.compare(str4) << endl;   // 0

// You can also use the operator == or !=
if (str1 == str2) cout << "Same" << endl;
if (str3 != str4) cout << "Different" << endl;
cout << boolalpha;  // print bool as true/false
cout << (str1 != str2) << endl;
cout << (str1 == str4) << endl;



Search/Replacing characters: You can use the functions available in the <algorithm> such as replace(). For example,

#include <algorithm>
......
string str("Code Quotient");
replace(str.begin(), str.end(), 'o', '_');
cout << str << endl;      // "C_de Qu_tient"



char name[100]={0};//string with space
  cout<<"Enter your name: ";
  cin.getline(name,100);



*/

void posiAlphaOrder(char ch)
{
  cout << (ch & 31);
}
void erase_eleV(vector<int> nums, int position)
{
  auto it = nums.begin() + position;
  nums.erase(it);

  for (auto it : nums)
  {
    cout << it << " -- ";
  }
  return;
}

vector<int> i_merge(vector<int> v1, vector<int> v2)
{
  vector<int> in(v1.size(), v2.size());
  merge(v1.begin(), v1.end(), v2.begin(), v2.end(), in.begin());
  return in;
}

void _replace(vector<int> vc, int me, int by)
{
  replace(vc.begin(), vc.end(), me, by);
}

bool find_Vec(vector<int> &v, int x)
{
  if (find(v.begin(), v.end(), x) != v.end())
    return true;
  return false;
}
int find_idxVC(vector<int> &v, int x)
{
  auto it = find(v.begin(), v.end(), x);
  if (it != v.end())
  {
    int idx = it - v.begin();
    return idx;
  }
  else
  {
    return -1;
  }
}

/*
must be in sort format :::

low=std::lower_bound (v.begin(), v.end(), 20); //          ^
  up= std::upper_bound (v.begin(), v.end(), 20); //                   ^

  std::cout << "lower_bound at position " << (low- v.begin()) << '\n';
  std::cout << "upper_bound at position " << (up - v.begin()) << '\n';
*/

// Traverse the vector in reverse order
/*
for (auto it = v.rbegin(); it != v.rend(); ++it) {
    cout << it->first << " " << it->second << endl;
}
*/

// find_map(m,30)<<endl;
bool find_map(map<int, int> &m, int x)
{
  if (m.find(x) != m.end())
    return true;
  return false;
} //<find_Vec(vc,30)<<endl;
bool find_set(set<int> &s, int x)
{
  return (s.find(x) != s.end());
}

/*.................................................................SET.........................................................................*/
vector<int> set_trv_inRange(set<int> &my_set, int start, int end)
{
  auto lower_bound_it = my_set.lower_bound(start);
  auto upper_bound_it = my_set.upper_bound(end);

  vector<int> st_v;
  for (auto it = lower_bound_it; it != upper_bound_it; ++it)
  {
    // cout << *it << " ";
    st_v.push_back(*it);
  }
  return st_v;
  /*
int count = 0;
auto it = my_set.begin();
while (it != my_set.end() && count < 4) {
   std::cout << *it << " ";
   ++it;
   ++count;
}


 for(auto it =st.begin(); it!=st.end(); it++){
    cout<<*it<<" ";
  }

*/
}
/*..............................................................List................................................................................*/
/*
List store data sequentially it work on the consept of doubble link list it ascess the random memory adress we can do many opersation on it ....
it having PUSH POP opertaion like deque
*/

/****************************************------- PREFIX  @@@ SUFIX --------*******************************************************/
int n;
vector<int> pr(n), sf(n);
void all_pre_suf(vector<int> nums)
{
  n = nums.size();

  int mx1 = nums[0], mx2 = nums[n - 1];
  for (int i = 0; i < n; i++)
  {
    pr[i] = mx1;
    mx1 = max(mx1, nums[i]);

    sf[n - 1 - i] = mx2;
    mx2 = max(mx2, nums[n - 1 - i]);
  }
}
vector<int> prefixsum(vector<int> vi)
{
  vector<int> res(vi.size());
  partial_sum(vi.begin(), vi.end(), res.begin());
  return res;
}
vector<int> sufixsum(vector<int> vi)
{
  vector<int> res(vi.size());
  reverse(vi.begin(), vi.end());
  partial_sum(vi.begin(), vi.end(), res.begin());
  return res;
}
/*Greatest element among the elements to its right otherwise -1 */
vector<int> replaceElements(vector<int> &a)
{
  vector<int> ns(a.size(), -1);
  int mx = a[a.size() - 1], p;
  for (int i = a.size() - 2; i >= 0; i--)
  {
    ns[i] = mx;
    mx = max(mx, a[i]);
  }
  return ns;
}

vector<int> near_leftSmaller(int n, int a[])
{

  vector<int> ans;
  stack<int> st;
  for (int i = 0; i < n; i++)
  {
    if (st.empty())
    {
      ans.push_back(-1);
    }
    else
    {
      while (!st.empty() and st.top() >= a[i])
      {
        st.pop();
      }
      if (!st.empty())
      {
        ans.push_back(st.top());
      }
      else
      {
        ans.push_back(-1);
      }
    }
    st.push(a[i]);
  }
  return ans;
}

/****************************************----------GRAPH--------*******************************************************/

/*
1.BFS   2. DFS   3. TOPO SORT 4.

*/
// directed acyclic graph to undirected acyclic graph  CREATED by TREE
/*nine
    void Tree_Grap(Node* root,unordered_map<int,vector<int>>& graph){
        if(root->left!=NULL){
            graph[root->data].push_back(root->left->data);
            graph[root->left->data].push_back(root->data);
            Tree_Grap(root->left,graph);
        }
        if(root->right!=NULL){
            graph[root->data].push_back(root->right->data);
            graph[root->right->data].push_back(root->data);
            Tree_Grap(root->right,graph);
        }
    }
*/

/*---------------------------------------------------------ALGO--------------------------------------------------------------------------*/
// Compute the longest proper prefix that is also a suffix of the pattern
vector<int> computeLPS(string pattern)
{
  int m = pattern.size();
  vector<int> lps(m, 0);
  int len = 0;
  int i = 1;
  while (i < m)
  {
    if (pattern[i] == pattern[len])
    {
      len++;
      lps[i] = len;
      i++;
    }
    else
    {
      if (len != 0)
      {
        len = lps[len - 1];
      }
      else
      {
        lps[i] = 0;
        i++;
      }
    }
  }
  return lps;
}

// Find all occurrences of the pattern in the text using KMP algorithm   T.C = O(n+m);
bool KMP(string text, string pattern)
{
  int n = text.size();
  int m = pattern.size();

  // Compute the longest proper prefix that is also a suffix of the pattern
  vector<int> lps = computeLPS(pattern);

  int i = 0; // index for text[]
  int j = 0; // index for pattern[]
  while (i < n)
  {
    if (pattern[j] == text[i])
    {
      j++;
      i++;
    }
    if (j == m)
    {
      return 1;
      cout << "Found pattern at index " << i - j << endl;
      j = lps[j - 1];
    }
    else if (i < n && pattern[j] != text[i])
    {
      if (j != 0)
      {
        j = lps[j - 1];
      }
      else
      {
        i++;
      }
    }
  }
  return 0;
}

/* Making cycalic graph in u-v-distace value*/
void U_V_Dis_toAdj(int n, vector<vector<int>> &edges)
{
  vector<vector<pair<int, int>>> graph(n + 1);
  for (auto &edge : edges)
  {
    graph[edge[0]].push_back({edge[1], edge[2]});
    graph[edge[1]].push_back({edge[0], edge[2]});
  }
}
void dfs(int node, vector<int> adj[], vector<int> &visited, stack<int> &s)
{
  visited[node] = 1;
  for (auto i : adj[node])
  {
    if (!visited[i])
    {
      dfs(i, adj, visited, s);
    }
    s.push(node);
  }
}

void calDFS(int x, int y, vector<vector<char>> &mat, int n, int m)
{
  if (x < 0 || x >= n || y < 0 || y >= m || mat[x][y] == 'X')
  {
    return;
  }

  if (mat[x][y] == 'O')
  {
    mat[x][y] = '.'; //! ONLY GO FOR DFS
    calDFS(x - 1, y, mat, n, m);
    calDFS(x + 1, y, mat, n, m);
    calDFS(x, y - 1, mat, n, m);
    calDFS(x, y + 1, mat, n, m);
    mat[x][y] = 'O'; // ! BACK TRACK
  }
}
/*
! [8 - direction- call]
return 1+caldfs(i+1,j-1,n,m,grid)+caldfs(i+1,j+1,n,m,grid)+caldfs(i-1,j-1,n,m,grid)
+caldfs(i-1,j+1,n,m,grid)+caldfs(i+1,j,n,m,grid)+caldfs(i,j-1,n,m,grid)
+caldfs(i-1,j,n,m,grid)+caldfs(i,j+1,n,m,grid);
*/

// BFS KHANS ALGO
vector<int> khan_topoSort(int V, vector<int> adj[])
{
  vector<int> topo;
  vector<int> indegree(V, 0);
  for (int i = 0; i < V; i++)
  {
    for (int u : adj[i])
    {
      indegree[u]++;
    }
  }
  queue<int> q;
  for (int i = 0; i < V; i++)
  {
    if (indegree[i] == 0)
    {
      q.push(i);
    }
  }
  while (!q.empty())
  {
    int top = q.front();
    q.pop();
    topo.push_back(top);
    for (int u : adj[top])
    {
      indegree[u]--;
      if (!indegree[u])
        q.push(u);
    }
  }
  return topo;
  // if(topo.size()==v) return true  // topo sort is valid on graph or not [ graph is sycalic or not]
}

vector<int> Dfs_topoSort(int V, vector<int> adj[])
{
  vector<int> visited(V);
  stack<int> s;
  for (int i = 0; i < V; i++)
  {
    if (!visited[i])
      dfs(i, adj, visited, s);
  }
  vector<int> ans;
  while (!s.empty())
  {
    ans.push_back(s.top());
    s.pop();
  }
  return ans;
}

class DisjointSet
{
  vector<int> rank, parent;

public:
  DisjointSet(int n)
  {
    rank.resize(n + 1, 0);
    parent.resize(n + 1);
    for (int i = 0; i <= n; i++)
    {
      parent[i] = i;
    }
  }

  int findUPar(int node)
  {
    if (node == parent[node])
      return node;
    return parent[node] = findUPar(parent[node]);
  }

  void unionByRank(int u, int v)
  {
    int ulp_u = findUPar(u);
    int ulp_v = findUPar(v);
    if (ulp_u == ulp_v)
      return;
    if (rank[ulp_u] < rank[ulp_v])
    {
      parent[ulp_u] = ulp_v;
    }
    else if (rank[ulp_v] < rank[ulp_u])
    {
      parent[ulp_v] = ulp_u;
    }
    else
    {
      parent[ulp_v] = ulp_u;
      rank[ulp_u]++;
    }
  }
  /*
  int main(){
  DisjointSet(7);
  ds.unionByRank(6, 7);
  ds.unionByRank(3, 2);
  // if 3 and 7 same or not
  if (ds.findUPar(3) == ds.findUPar(7)) {
      cout << "Same\n";
  }
  }
  */
};

int main()
{
  /*
 --COMENT---
 ! ALERT
 * impotent u
 ? dout me
  TODO : GO IT
 */

  set<int> my_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vector<int> vc = set_trv_inRange(my_set, 0, 4);
  for (auto it : vc)
  {
    cout << it << " ";
  }
  cout << endl;
  cout << KMP("ABCGIKOP", "KOP");
  cout << __CountBit_n(78);
  return 0;
}