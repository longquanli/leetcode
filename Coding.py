# List
# 2 Add Two Numbers
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        carry = 0
        head = dummy
        
        while l1 or l2:
            if l1 and l2:
                val = (l1.val + l2.val + carry) % 10
                carry = (l1.val + l2.val + carry) // 10
                l1 = l1.next
                l2 = l2.next
            elif l1:
                val = (l1.val + carry) % 10
                carry = (l1.val + carry) // 10
                l1 = l1.next
            else:
                val = (l2.val + carry) % 10
                carry = (l2.val + carry) // 10
                l2 = l2.next
            head.next = ListNode(val)
            head = head.next
        
        if carry != 0:
            head.next = ListNode(carry)

        return dummy.next
# 445
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        stack1 = []
        stack2 = []
        
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        
        carry = 0
        sumArr = []

        while stack1 or stack2:
            if stack1 and stack2:
                val = stack1.pop() + stack2.pop() + carry
            elif stack1:
                val = stack1.pop() + carry
            else:
                val = stack2.pop() + carry
            digit = val % 10
            carry = val // 10
            sumArr.append(digit)
        
        if carry != 0:
            sumArr.append(carry)

        reverseArr = self.reverse(sumArr)

        return self.makeLinkedList(reverseArr)
    
    def reverse(self, arr):
        start = 0
        end = len(arr) - 1
        while start <= end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
        return arr

    def makeLinkedList(self, arr):
        dummy = ListNode(0)
        head = dummy

        for digit in arr:
            head.next = ListNode(digit)
            head = head.next
        
        return dummy.next

# 24 Swap Nodes in Paris
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        preNode = dummy

        while head and head.next:
            nextNode = head.next
            preNode.next = nextNode
            head.next = nextNode.next
            nextNode.next = head
            preNode = head
            head = head.next
        
        return dummy.next


# 206 Reverse LinkedList
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        preNode = None

        while head:
            nextNode = head.next
            head.next = preNode
            preNode = head
            head = nextNode
        
        return preNode

# 21 Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(0)
        head = dummy

        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next

        if l1:
            head.next = l1
        if l2:
            head.next = l2

        return dummy.next

# 23 Merge k Sorted Lists
# Onlgk On
class Solution:
    def mergeKLists(self, lists):
        pq = []
        for i in range(len(lists)):
            node = lists[i]
            if node is None:
                continue
            heapq.heappush(pq, (node.val, i, node))

        dummy = ListNode(0)
        head = dummy

        while pq:
            value, _, idx, node = heapq.heappop(pq)
            head.next = ListNode(value)
            head = head.next
            if node.next:
                heapq.heappush(pq, (node.next.val, idx, node.next))

        return dummy.next

# 147 Insertion Sort List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)

        while head:
            node = dummy
            while node.next and node.next.val < head.val:
                node = node.next
            nxt = head.next
            head.next = node.next
            node.next = head
            head = nxt
        
        return dummy.next

# 148 Sort List
# Version I Onlgn Olgn
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head

        midNode = self.findMidNode(head)
        rightHead = midNode.next
        midNode.next = None
        leftNode = self.sortList(head)
        rightNode = self.sortList(rightHead)

        return self.mergeTwoLists(leftNode, rightNode)

    def findMidNode(self, head):
        slow = head
        fast = head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    def mergeTwoLists(self, left, right):
        dummy = ListNode(0)
        head  = dummy

        while left and right:
            if left.val < right.val:
                head.next = left
                left = left.next
            else:
                head.next = right
                right = right.next
            head = head.next
        
        if left:
            head.next = left
        if right:
            head.next = right

        return dummy.next

# Version II iteration
class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head

        count = self.getCount(head)
        start = head
        dummy = ListNode(0)
        
        for size in range(0, count, 2):
            tail = dummy
            while start:
                if start.next is None:
                    tail.next = start
                    break
                mid, nxtSubList = self.split(start, size)
                self.merge(start, mid)
                start = nxtSubList
            start = dummy.next

        return dummy.next

    def split(self, head, size):
        slow = head
        fast = head.next
        counter = 0

        while fast and fast.next:
            if counter >= size:
                break
            fast = fast.next.next
            slow = slow.next
            counter += 1

        mid = slow.next
        slow.next = None
        nextSubList = fast.next
        fast.next = None
        
        return mid, nextSubList

    def mergeTwoLists(self, left, right):
        dummy = ListNode(0)
        head  = dummy

        while left and right:
            if left.val < right.val:
                head.next = left
                left = left.next
            else:
                head.next = right
                right = right.next
            head = head.next
        
        if left:
            head.next = left
        if right:
            head.next = right

        return dummy.next

    def getCount(self, head):
        counter = 0
        while head:
            head = head.next
            counter += 1

        return counter

# 707 Design Linked List
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None

class MyLinkedList:

    def __init__(self):
         self.head = ListNode(-1)
         self.tail = ListNode(-1)
         self.head.next = self.tail
         self.tail.prev = self.head


    def get(self, index: int) -> int:
        head = self.head

        for _ in range(index+1):
            if head is None:
                return -1
            head = head.next
        
        if head is None:
            return -1
            
        return head.val

    def addAtHead(self, val: int) -> None:
        node = ListNode(val)
        firstNode = self.head.next
        node.prev = self.head
        node.next = firstNode
        firstNode.prev = node
        self.head.next = node

    def addAtTail(self, val: int) -> None:
        node = ListNode(val)
        lastNode = self.tail.prev
        node.next = self.tail
        node.prev = lastNode
        lastNode.next = node
        self.tail.prev = node

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0:
            return

        node = ListNode(val)
        prevNode = self.head
        
        for _ in range(index):
            if prevNode is None:
                return
            prevNode = prevNode.next
        
        if prevNode is None or prevNode == self.tail:
            return
        
        nextNode = prevNode.next
        node.prev = prevNode
        node.next = nextNode
        prevNode.next = node
        nextNode.prev = node

    def deleteAtIndex(self, index: int) -> None:
        if index < 0:
            return
        prevNode = self.head

        for _ in range(index):
            if prevNode is None:
                return
            prevNode = prevNode.next

        if prevNode is None or prevNode.next is None or prevNode == self.tail or prevNode.next == self.tail:
            return

        nextNode = prevNode.next.next
        prevNode.next = nextNode
        nextNode.prev = prevNode
        


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)

# 125 Valid Palindrome
class Solution:
    def isPalindrome(self, s):
        if len(s) <= 1:
            return True

        start = 0
        end = len(s) - 1

        while start < end:
            while start < end and not s[start].isalpha() and not s[start].isdigit():
                start += 1
            while start < end and not s[end].isalpha() and not s[end].isdigit():
                end -= 1
            if s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1

        return True

# 455 Assign Cookies
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        counter = 0
        idx = 0
        
        for content in g:
            while idx < len(s) and s[idx] < content:
                idx += 1
            if idx == len(s):
                break
            idx += 1
            counter += 1
        
        return counter

# 11 Container With Most Water
class Solution:
    def maxArea(self, height):
        maxAr = 0
        left = 0
        right = len(height) - 1

        while left < right:
            maxAr = max(maxAr, self.getArea(left, right, min(height[left], height[right])))
            if height[left] < height[right]:
                while left < right and height[left] >= height[left + 1]:
                    left += 1
                left += 1
            else:
                while left < right and height[right] >= height[right - 1]:
                    right -= 1
                right -= 1
            maxAr = max(maxAr, self.getArea(left, right, min(height[left], height[right])))

        return maxAr

    def getArea(self, left, right, height):
        return (right - left) * height
# 42 Trapping Rain Water
class Solution:
    def trap(self, height: List[int]) -> int:
        water = 0
        left = 0
        right = len(height) - 1
        leftHeightMax = height[left]
        rightHeightMax = height[right]

        while left <= right:
            if leftHeightMax < rightHeightMax:
                leftHeightMax = max(leftHeightMax, height[left])
                water += leftHeightMax - height[left]
                left += 1
            else:
                rightHeightMax = max(rightHeightMax, height[right])
                water += rightHeightMax - height[right]
                right -= 1

        return water

# 917 reverse only letters
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        left = 0
        right = len(s) - 1
        stringList = list(s)

        while left < right:
            while left < right and not stringList[left].isalpha():
                left += 1
            while left < right and not stringList[right].isalpha():
                right -= 1
            if left < right:
                stringList[left], stringList[right] = stringList[right], stringList[left]
            left += 1
            right -= 1
        
        return "".join(stringList)

# 925 Long pressed name
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        pointer1 = 0
        pointer2 = 0

        while pointer1 < len(name) and pointer2 < len(typed):
            char1 = name[pointer1]
            char2 = typed[pointer2]
            if char1 == char2:
                pointer1 += 1
                pointer2 += 1
            elif pointer2 != 0 and char2 == typed[pointer2 - 1]:
                pointer2 += 1
            else:
                return False
        
        if pointer1 != len(name):
            return False
        
        while pointer2 < len(typed):
            if pointer2 != 0 and typed[pointer2] != typed[pointer2 - 1]:
                return False
            pointer2 += 1
        
        return True

# 986 interval list intersections
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        pointer1 = 0
        pointer2 = 0
        overlappedLists = []

        while pointer1 < len(firstList) and pointer2 < len(secondList):
            start1, end1 = firstList[pointer1]
            start2, end2 = secondList[pointer2]
            overlappedList = self.getOverlappedList(start1, end1, start2, end2)
            if len(overlappedList) != 0:
                overlappedLists.append(overlappedList)
            if end1 > end2:
                pointer2 += 1
            elif end1 < end2:
                pointer1 += 1
            else:
                pointer1 += 1
                pointer2 += 1
        
        return overlappedLists
    
    def getOverlappedList(self, start1, end1, start2, end2):
        if start1 > end2 or start2 > end1:
            return []
        return [max(start1, start2), min(end1, end2)]

# 881 boats to save people
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        if len(people) == 0:
            return 0
            
        people.sort()
        start = 0
        end = len(people) - 1
        ships = 0

        while start < end:
            if people[end] > limit:
                return -1
            elif people[end] == limit:
                ships += 1
                end -= 1
            elif people[end] + people[start] > limit:
                ships += 1
                end -= 1
            else:
                ships += 1
                start += 1
                end -= 1
        
        if start == end:
            ships += 1

        return ships

# 167 two sum II
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            cur = numbers[left] + numbers[right]
            if cur == target:
                return [left + 1, right + 1]
            elif cur < target:
                left += 1
            else:
                right -= 1
        
        return []

# 3 sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        combinations = []
        nums.sort()

        for i in range(len(nums) - 2):
            # remove duplicats
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                cur = nums[i] + nums[left] + nums[right]
                if cur == 0:
                    combinations.append([nums[i], nums[left], nums[right]])
                    # remove duplicats
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    left += 1
                    # remove duplicats
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    right -= 1
                elif cur < 0:
                    left += 1
                else:
                    right -= 1
        
        return combinations

# 3 sum closest
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        curSum = float('inf')
        nums.sort()

        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1

            while left < right:
                cur = nums[i] + nums[left] + nums[right]
                if abs(cur - target) < abs(curSum - target):
                    curSum = cur
                if cur > target:
                    right -= 1
                else:
                    left += 1
        
        return curSum
                

# 977 squares of sorted array
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        index = self.findIndexCloseToZeroValue(nums)

        left = index
        right = index + 1
        squareList = []

        while left >= 0 and right < len(nums):
            leftVal = nums[left] ** 2
            rightVal = nums[right] ** 2
            if leftVal < rightVal:
                squareList.append(leftVal)
                left -= 1
            else:
                squareList.append(rightVal)
                right += 1
        
        while left >= 0:
            squareList.append(nums[left] ** 2)
            left -= 1
        
        while right < len(nums):
            squareList.append(nums[right] ** 2)
            right += 1
        
        return squareList

    def findIndexCloseToZeroValue(self, nums):
        index = 0
        cur = float('inf')

        for i in range(len(nums)):
            if abs(nums[i]) < abs(nums[index]):
                index = i
        
        return index

# 992 subarrays with k different integers
class Solution:
    def subarrayWithDistinct(self, nums, k):
        return self.findAtMostK(nums, k) - self.findAtMostK(nums, k - 1)

    def findAtMostK(self, nums, k):
        memo = collections.defaultdict(int)
        left = 0
        distincts = 0

        for right in range(len(nums)):
            rightVal = nums[right] 
            memo[rightVal] += 1
            while len(memo) > k:
                leftVal = nums[left]
                memo[leftVal] -= 1
                if memo[leftVal] == 0:
                    memo.pop(leftVal)
                left += 1
            distincts += (right - left + 1)

        return distincts

# 296 Best Meeting Point
# O(n*m*n*m) O(n*m)
class Solution:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def minTotalDistance(self, grid: List[List[int]]) -> int:
        distances = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (grid[i][j] == 0):
                    continue
                self.updateDistances(i, j, grid, distances)
        
        startI, startJ = 0, 0
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if (distances[i][j] < distances[startI][startJ]):
                    startI, startJ = i, j
        
        return distances[startI][startJ]
    
    def updateDistances(self, startI, startJ, grid, distances):
        queue = collections.deque([(startI, startJ)])
        visited = set([(startI, startJ)])
        m, n = len(grid), len(grid[0])
        distance = 0

        while queue:
            size = len(queue)
            distance += 1
            for _ in range(size):
                curI, curJ = queue.popleft()
                for direction in self.dirs:
                    nextI, nextJ = curI + direction[0], curJ + direction[1]
                    if not self.isPositionValid(nextI, nextJ, m, n, visited):
                        continue
                    queue.append((nextI, nextJ))
                    distances[nextI][nextJ] += distance
                    visited.add((nextI, nextJ))
        
    
    def isPositionValid(self, posI, posJ, rows, columns, visited):
        if posI < 0 or posI >= rows or posJ < 0 or posJ >= columns or (posI, posJ) in visited:
            return False
        return True

# O(n*m) O(max(m, n))
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        homeInRow = []
        homeInCol = []

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 0:
                    continue
                homeInRow.append(i)
                homeInCol.append(j)
        
        homeInCol.sort()
        return self.getDistance(homeInRow, homeInRow[len(homeInRow) // 2]) + self.getDistance(homeInCol, homeInCol[len(homeInCol) // 2])

    def getDistance(self, positions, startPoint):
        distance = 0
        for position in positions:
            distance += abs(position - startPoint)
        return distance

# 133 Clone Graph
class Solution:
    def cloneGraph(self, node):
        if node is None:
            return None

        nodes = self.getNodes(node)
        graph = self.cloneNodes(nodes)
        self.connectNodes(graph, nodes)

        return graph[node]

    def getNodes(self, node):
        visited = set([node])
        queue = collections.deque([node])

        while queue:
            cur = queue.popleft()
            for neighbor in cur.neighbors:
                if neighbor in visited:
                    continue
                queue.append(neighbor)
                visited.add(neighbor)

        return visited

    def cloneGraph(self, nodes):
        graph = dict()
        for node in nodes:
            newNode = Node(node.val)
            graph[node] = newNode

        return graph

    def connectNodes(self, graph, nodes):
        for node in nodes:
            newNode = graph[node]
            for neighbor in node.neighbors:
                newNeighbor = graph[neighbor]
                newNode.neighbors.append(newNeighbor)

# 138 Copy List with Random Pointer
class Node:
    def __init__(self, value, next = None, random = None):
        self.val = value
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head):
        if head is None:
            return None

        nodes = self.getNodes(head)
        graph = self.cloneGraph(nodes)
        self.connectNodes(head, graph)

        return graph[head]

    def getNodes(self, head):
        nodes = set()

        while head:
            nodes.add(head)
            head = head.next

        return nodes

    def cloneGraph(self, nodes):
        graph = dict()

        for node in nodes:
            newNode = Node(node.val)
            graph[node] = newNode

        return graph

    def connectNodes(self, head, graph):
        while head:
            newHead = graph[head]
            newNxtNode = graph[head.next] if head.next in graph else None
            newRandomNode = graph[head.random] if head.random in graph else None
            newHead.next = newNxtNode
            newHead.random = newRandomNode
            head = head.next

# 200 Number of Islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == "0":
                    continue
                self.bfs(grid, i, j)
                islands += 1
        
        return islands
    
    def bfs(self, grid, startR, startC):
        queue = collections.deque([(startR, startC)])

        while queue:
            curR, curC = queue.popleft()
            for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxtR, nxtC = curR + direction[0], curC + direction[1]
                if not self.isPositionValid(grid, nxtR, nxtC):
                    continue
                queue.append((nxtR, nxtC))
                grid[nxtR][nxtC] = "0"
    
    def isPositionValid(self, grid, nxtR, nxtC):
        if nxtR < 0 or nxtR >= len(grid) or nxtC < 0 or nxtC >= len(grid[nxtR]):
            return False
        return grid[nxtR][nxtC] == "1"

# 547 Number of provinces
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        graph = self.buildGraph(isConnected)
        provinces = 0
        visited = set()

        for i in range(len(isConnected)):
            if  i in visited:
                continue
            self.bfs(graph, visited, i)
            provinces += 1
        
        return provinces
    
    def buildGraph(self, isConnected):
        graph = collections.defaultdict(list)
        
        for i in range(len(isConnected)):
            for j in range(len(isConnected[i])):
                if isConnected[i][j]:
                    graph[i].append(j)
                    graph[j].append(i)
        
        return graph
    
    def bfs(self, graph, visited, start):
        queue = collections.deque([start])
        visited.add(start)

        while queue:
            cur = queue.popleft()
            for nxt in graph[cur]:
                if nxt in visited:
                    continue
                queue.append(nxt)
                visited.add(nxt)

# 695 Max area of island
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        maxArea = 0
        visited = set()

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 0:
                    continue
                curArea = self.bfs(grid, i, j, visited)
                maxArea = max(maxArea, curArea)
        
        return maxArea
    
    def bfs(self, grid, startR, startC, visited):
        queue = collections.deque([(startR, startC)])
        visited.add((startR, startC))
        area = 0 # initial 0

        while queue:
            cur = queue.popleft()
            for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxtR, nxtC = direction[0] + cur[0], direction[1] + cur[1]
                if not self.isPositionValid(grid, visited, nxtR, nxtC):
                    continue
                queue.append((nxtR, nxtC))
                visited.add((nxtR, nxtC))
            area += 1
        
        return area
    
    def isPositionValid(self, grid, visited, r, c):
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[r]):
            return False
        if (r, c) in visited:
            return False
        return grid[r][c] == 1

# 733 Flood Fill
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if image[sr][sc] == color:
            return image
        
        queue = collections.deque([(sr, sc)])
        originColor = image[sr][sc]
        image[sr][sc] = color

        while queue:
            cur = queue.popleft()
            for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxtR, nxtC = cur[0] + direction[0], cur[1] + direction[1]
                if not self.isPositionValid(image, originColor, nxtR, nxtC):
                    continue
                queue.append((nxtR, nxtC))
                image[nxtR][nxtC] = color
        
        return image
    
    def isPositionValid(self, image, color, r, c):
        if r < 0 or r >= len(image) or c < 0 or c >= len(image[r]):
            return False
        if image[r][c] != color:
            return False
        return True

# 1162 As Far from Land as Possible
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        distance = dict()

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 0:
                    continue
                if (i, j) in distance:
                    continue
                self.bfs(distance, grid, i, j)
        
        maxDistance = -1
        for position in distance:
            # Island is not counted
            if distance[position] == 0:
                continue
            maxDistance = max(maxDistance, distance[position])
        
        return maxDistance
    
    def bfs(self, distance, grid, r, c):
        distance[(r, c)] = 0
        queue = collections.deque([(r, c)])
        step = 0

        while queue:
            size = len(queue)
            step += 1
            for _ in range(size):
                cur = queue.popleft()
                for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nxtR, nxtC = cur[0] + direction[0], cur[1] + direction[1]
                    if not self.isPositionValid(distance, grid, nxtR, nxtC, step):
                        continue
                    queue.append((nxtR, nxtC))
                    distance[(nxtR, nxtC)] = step
        
    def isPositionValid(self, distance, grid, r, c, step):
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[r]):
            return False
        if grid[r][c] == 1:
            return False
        if (r, c) in distance and distance[(r, c)] <= step:
            return False
        return True

# 827 Making A Large Island

# 841 Keys and Rooms
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = set([0])
        queue = collections.deque([0])

        while queue:
            cur = queue.popleft()
            for nxt in rooms[cur]:
                if nxt in visited:
                    continue
                queue.append(nxt)
                visited.add(nxt)
        
        return len(visited) == len(rooms)

# 1202 Smallest String With Swaps ****
class Solution:
    def smallestStringWithSwaps(self, s)

# 207 Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph, graphCounter = self.buildGraph(prerequisites)
        numOfCoursesTake = self.takeCourse(graph, graphCounter, numCourses)
        return numOfCoursesTake == numCourses
    
    def buildGraph(self, prerequisites):
        graph = collections.defaultdict(list)
        graphCounter = collections.defaultdict(int)

        for pair in prerequisites:
            course, preCourse = pair[0], pair[1]
            graph[pair[1]].append(pair[0])
            graphCounter[pair[0]] += 1
        
        return (graph, graphCounter)
    
    def takeCourse(self, graph, graphCounter, numCourses):
        queue = collections.deque()
        numOfCoursesTake = 0

        for course in range(numCourses):
            if graphCounter[course] != 0:
                continue
            queue.append(course)
            numOfCoursesTake += 1
        
        while queue:
            course = queue.popleft()
            for nxtCourse in graph[course]:
                graphCounter[nxtCourse] -= 1
                if graphCounter[nxtCourse] == 0:
                    queue.append(nxtCourse)
                    numOfCoursesTake += 1
        
        return numOfCoursesTake

# 210
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph, graphCounter = self.buildGraph(prerequisites)
        coursesTaken = self.takeCourse(graph, graphCounter, numCourses)
        return coursesTaken if len(coursesTaken) == numCourses else []
    
    def buildGraph(self, prerequisites):
        graph = collections.defaultdict(list)
        graphCounter = collections.defaultdict(int)

        for pair in prerequisites:
            course, preCourse = pair[0], pair[1]
            graph[pair[1]].append(pair[0])
            graphCounter[pair[0]] += 1
        
        return (graph, graphCounter)
    
    def takeCourse(self, graph, graphCounter, numCourses):
        queue = collections.deque()
        coursesTaken = []

        for course in range(numCourses):
            if graphCounter[course] != 0:
                continue
            queue.append(course)
            coursesTaken.append(course)
        
        while queue:
            course = queue.popleft()
            for nxtCourse in graph[course]:
                graphCounter[nxtCourse] -= 1
                if graphCounter[nxtCourse] == 0:
                    queue.append(nxtCourse)
                    coursesTaken.append(nxtCourse)
        
        return coursesTaken

# 802 Find Eventual Safe States
class Solution:
    def eventualSafeNodes(self, graph):
        safeNodes = []
        memo = dict()
        newGraph = self.buildGraph(graph)

        for startNode in range(len(graph)):
            if self.dfs(startNode, memo, set(), newGraph):
                safeNodes.append(startNode)

        return safeNodes

    def buildGraph(self, connection):
        graph = dict()

        for node in range(len(connection)):
            graph[node] = []
            for connectedNode in connection[node]:
                graph[node].append(connectedNode)

        return graph

    def dfs(self, startNode, memo, visited, graph):
        if startNode in memo:
            return memo[startNode]

        for nxtNode in graph[startNode]:
            if nxtNode in visited:
                memo[startNode] = False
                return False
            visited.add(nxtNode)
            find = self.dfs(nxtNode, memo, visited, graph)
            if not find:
                memo[startNode] = False
                return False
            visited.remove(nxtNode)


        memo[startNode] = True
        return True

# 990 Satisfiability of Equality Equations
class Solution:
    def equationsPossible(self, equations):
        equal = collections.defaultdict(list)
        for equation in equations:
            if equation[1:3] != "==":
                continue
            equal[equation[0]].append(equation[-1])
            equal[equation[-1]].append(equation[0])

        memo = set()
        for equation in equations:
            if equation[1:3] == "!=":
                if equation[0] == equation[-1]:
                    return False
                if self.canConnect(equation[0], equation[-1], set(equation[0]), equal, memo):
                    return False
        return True

    def canConnect(self, start, end, visited, equal, memo):
        if start == end:
            return True
        if (start + end) in memo:
            return False
        for nxt in equal[start]:
            if nxt in visited:
                continue
            visited.add(nxt)
            if self.canConnect(nxt, end, visited, equal, memo):
                return True
            visited.remove(nxt)

        memo.add(start+end)
        return False

# 721
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        nameToEmails, emailToEmails = self.buildAccountGraph(accounts)
        visited = set()
        mergedAccounts = []

        for name in nameToEmails.keys():
            for email in nameToEmails[name]:
                if email in visited:
                    continue
                mergedEmails = self.bfs(email, emailToEmails, visited)
                mergedAccount = [name] + sorted(mergedEmails)
                mergedAccounts.append(mergedAccount)
        
        return mergedAccounts
    
    def buildAccountGraph(self, accounts):
        nameToEmails = collections.defaultdict(list)
        emailToEmails = collections.defaultdict(list)
        
        for account in accounts:
            name = account[0]
            emails = account[1:]
            baseEmail = account[1]
            for email in emails:
                emailToEmails[baseEmail].append(email)
                emailToEmails[email].append(baseEmail)
                nameToEmails[name].append(email)
            
        return (nameToEmails, emailToEmails)

    def bfs(self, email, emailToEmails, visited):
        queue = collections.deque([email])
        visited.add(email)
        emails = [email]

        while queue:
            curEmail = queue.popleft()
            for nxtEmail in emailToEmails[curEmail]:
                if nxtEmail in visited:
                    continue
                queue.append(nxtEmail)
                visited.add(nxtEmail)
                emails.append(nxtEmail)
        
        return emails
# 737 Sentence Similarity II
# No memo
class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        if len(sentence1) != len(sentence2):
            return False
        
        wordGraph = self.buildWordGraph(similarPairs)
        
        for i in range(len(sentence1)):
            if not self.isSimilar(sentence1[i], sentence2[i], wordGraph, set([sentence1[i]])):
                return False
        
        return True
    
    def buildWordGraph(self, pairs):
        graph = collections.defaultdict(list)

        for pair in pairs:
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])
        
        return graph
    
    def isSimilar(self, startWord, endWord, graph, visited):
        if startWord == endWord:
            return True
        
        for nxtWord in graph[startWord]:
            if nxtWord in visited:
                continue
            visited.add(nxtWord)
            if self.isSimilar(nxtWord, endWord, graph, visited):
                return True
            visited.remove(nxtWord)

        return False

# 952

# 886 Possible Bipartition
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        
        for dislike in dislikes:
            graph[dislike[0]].append(dislike[1])
            graph[dislike[1]].append(dislike[0])
        
        color = dict()
        for node in range(1, n+1):
            if node not in color:
                if not self.dfs(node, "b", graph, color):
                    return False
        
        return True
    
    def dfs(self, node, curColor, graph, color):
        if node in color:
            return color[node] == curColor
        color[node] = curColor
        for nxtNode in graph[node]:
            if curColor == "b":
                if not self.dfs(nxtNode, "r", graph, color):
                    return False
            else:
                if not self.dfs(nxtNode, "b", graph, color):
                    return False
        return True

# 1042 Flower Planting With No Adjacent ****
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        colors = [1,2,3,4]
        flowerColors = [-1] * n
        graph = self.buildGraph(paths, n)
        visited = dict()

        for i in range(1, n + 1):
            if i in visited:
                continue
            for color in colors:
                if self.findColor(i, color, colors, graph, visited):
                    continue

        for garden in visited.keys():
            flowerColors[garden - 1] = visited[garden]
    
        return flowerColors

    def buildGraph(self, paths, n):
        graph = collections.defaultdict(list)

        for path in paths:
            graph[path[0]].append(path[1])
            graph[path[1]].append(path[0])
        
        for i in range(1, n + 1):
            if i in graph:
                continue
            graph[i] = []
        
        return graph
    
    def findColor(self, garden, color, colors, graph, visited):
        if garden in visited:
            return visited[garden] == color
        
        visited[garden] = color
        for nxtGarden in graph[garden]:
            for nxtColor in colors:
                if self.findColor(nxtGarden, nxtColor, colors, graph, visited):
                    return True
        visited.pop(garden)
        
        return True

# 22 Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        parenthesis = []
        self.generateParenthesisImpl(0, n, parenthesis, [])
        
        return parenthesis
    
    def generateParenthesisImpl(self, leftCount, totalCount, parenthesis, path):
        if leftCount < 0 or totalCount < 0:
            return
        if totalCount == 0 and leftCount == 0:
            parenthesis.append("".join(path))
            return
        
        self.generateParenthesisImpl(leftCount + 1, totalCount - 1, parenthesis, path + ["("])
        self.generateParenthesisImpl(leftCount - 1, totalCount, parenthesis, path + [")"])

# 661 Image Smoother
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        smoothedImg = [[0 for _ in range(len(img[0]))] for _ in range(len(img))]

        for i in range(len(img)):
            for j in range(len(img[i])):
                count, total = self.calculate(i, j, img)
                smoothedImg[i][j] = total // count
        
        return smoothedImg
    
    def calculate(self, row, col, img):
        total = img[row][col]
        count = 1
        if row > 0:
            count += 1
            total += img[row - 1][col]
        if row < len(img) - 1:
            count += 1
            total += img[row + 1][col]
        if col > 0:
            count += 1
            total += img[row][col - 1]
        if col < len(img[row]) - 1:
            count += 1
            total += img[row][col + 1]
        if row > 0 and col > 0:
            count += 1
            total += img[row - 1][col - 1]
        if row > 0 and col < len(img[row]) - 1:
            count += 1
            total += img[row - 1][col + 1]
        if row < len(img) - 1 and col > 0:
            count += 1
            total += img[row + 1][col - 1]
        if row < len(img) - 1 and col < len(img[row]) - 1:
            count += 1
            total += img[row + 1][col + 1]
        
        return (count, total)

# 37 Sudoku Solver
class Solution:
    def solveSudoku(self, board):
        self.solver(board)

    def solver(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != ".":
                    continue
                for number in "123456789":
                    if not self.isValid(board, i, j, number):
                        continue
                    board[i][j] = number
                    if self.solver(board):
                        return True
                    board[i][j] = "."
                return False
        return True

    def isValid(self, board, row, col, number):
        for i in range(len(board[row])):
            if board[row][i] == number:
                return False

        for i in range(len(board)):
            if board[i][col] == number:
                return False

        for i in range(len(board)):
            if board[row // 3 * 3 + i // 3][col // 3 * 3 + i % 3] == number:
                return False

        return True

# 79 Word Search
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == word[0]:
                    if self.findWord(i, j, board, word, 1, set([(i, j)])):
                        return True
        
        return False
    
    def findWord(self, row, col, board, word, position, visited):
        if position == len(word):
            return True
        
        for nxtRow, nxtCol in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if not self.isValid(nxtRow, nxtCol, visited, board, word[position]):
                continue
            visited.add((nxtRow, nxtCol))
            if self.findWord(nxtRow, nxtCol, board, word, position + 1, visited):
                return True
            visited.remove((nxtRow, nxtCol))
        
        return False
    
    def isValid(self, row, col, visited, board, charactor):
        if row < 0 or col < 0 or row >= len(board) or col >= len(board[row]):
            return False
        if (row, col) in visited:
            return False
        return board[row][col] == charactor

# 80 Word Search II
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.word = ""

    def getWord(self):
        return self.word

    def setWord(self, word):
        self.word = word

    def getChildren(self):
        return self.children

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root

        for ch in word:
            children = node.getChildren()
            if ch not in children:
                children[ch] = TrieNode()
            node = children[ch]
        node.setWord(word)

    def getRoot(self):
        return self.root

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()

        for word in words:
            trie.insert(word)

        result = set()
        for i in range(len(board)):
            for j in range(len(board[i])):
                self.findWordsImpl(i, j, board, trie.getRoot(), set(), result)

        return list(result)
    
    def findWordsImpl(self, row, col, board, node, visited, result):
        if row < 0 or col < 0 or row >= len(board) or col >= len(board[row]):
            return

        if (row, col) in visited:
            return

        ch = board[row][col]
        children = node.getChildren()
        if ch not in children:
            return
        word = children[ch].getWord()
        if word != "":
            result.add(word)
        for nxtR, nxtC in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
            visited.add((row, col))
            self.findWordsImpl(nxtR, nxtC, board, children[ch], visited, result)
            visited.remove((row, col))

# Netflix
# 1578 Minimum changes to make alternating binary string
class Solution:
    def minOperations(self, s: str) -> int:
        return min(self.calculateChanges("1", s), self.calculateChanges("0", s))
    
    def calculateChanges(self, startingValue, s):
        count = 0
        tmpValue = startingValue

        for ch in s:
            if ch != tmpValue:
                count += 1
            tmpValue = "1" if tmpValue == "0" else "0"
        
        return count

# 20 Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        for ch in s:
            if ch == "(":
                stack.append(")")
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            else:
                if len(stack) == 0:
                    return False
                if stack[-1] != ch:
                    return False
                stack.pop()
        
        return len(stack) == 0

# 49 Group anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groupedAnagramsDict = collections.defaultdict(list)

        for string in strs:
            sortedString = "".join(sorted(string))
            groupedAnagramsDict[sortedString].append(string)
        
        groupedAnagrams = []
        for _, value in groupedAnagramsDict.items():
            groupedAnagrams.append(value)
        
        return groupedAnagrams

# 56 Merged Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0:
            return []
        
        mergedIntervals = []
        pq = []
        for interval in intervals:
            heapq.heappush(pq, interval)
        
        start, end = float('inf'), float('inf')
        
        while pq:
            interval = heapq.heappop(pq)
            if start == float('inf'):
                start = interval[0]
                end = interval[1]
                continue
            if end < interval[0]:
                mergedIntervals.append([start, end])
                start, end = interval[0], interval[1]
            else:
                end = max(end, interval[1])
            
        mergedIntervals.append([start, end])
        return mergedIntervals

# 41 First Missing Positive
class Solution:
    def firstMissingPositive(self, nums):
        index = 0
        while index < len(nums):
            num = nums[index]
            # out of boundary
            if num <= 0 or num > len(nums):
                index += 1
                continue
            # right position
            elif num == index + 1:
                index += 1
                continue
            # no need to do switch, same already
            elif num == nums[num - 1]:
                index += 1
                continue
            else:
                nums[index] = nums[num - 1]
                nums[num - 1]  num

        for i in range(len(nums)):
            if nums[i] != i + 1:
                return i + 1

        return len(nums) + 1

# 146 LRU Cache
class DoubleLinkedListNode:
    def __init__(self, val = float('-inf'), key = float('-inf')):
        self.val = val
        # key is needed.
        self.key = key
        self.next = None
        self.prev = None

class DoubleLinkedList:
    def __init__(self):
        self.head = DoubleLinkedListNode()
        self.tail = DoubleLinkedListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def add(self, node):
        nextNode = self.head.next
        node.prev = self.head
        node.next = nextNode
        nextNode.prev = node
        self.head.next = node
    
    def pop(self):
        prevNode = self.tail.prev
        if prevNode == self.head:
            return
        return self.remove(prevNode)
    
    def remove(self, node):
        prevNode = node.prev
        nextNode = node.next
        prevNode.next = nextNode
        nextNode.prev = prevNode
        node.next = None
        node.prev = None
        return node
    
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.curCapacity = 0
        self.keyNodeDict = dict()
        self.doubleLinkedList = DoubleLinkedList()

    def get(self, key: int) -> int:
        if key not in self.keyNodeDict:
            return -1
        node = self.keyNodeDict[key]
        self.doubleLinkedList.remove(node)
        self.doubleLinkedList.add(node)
        
        return node.val


    def put(self, key: int, value: int) -> None:
        if key in self.keyNodeDict:
            node = self.keyNodeDict[key]
            node.val = value
            self.doubleLinkedList.remove(node)
            self.doubleLinkedList.add(node)
            return
        
        node = DoubleLinkedListNode(value, key)
        self.doubleLinkedList.add(node)
        self.keyNodeDict[key] = node

        if (self.curCapacity < self.capacity):
            self.curCapacity += 1
            return

        removedNode = self.doubleLinkedList.pop()
        self.keyNodeDict.pop(removedNode.key)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


# 341 Flatten Nested List Iterator
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """
# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
# Version I
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.list = self.__flattenList(nestedList)
        self.index = 0
    
    def next(self) -> int:
        value = self.list[self.index]
        self.index += 1
        return value
    
    def hasNext(self) -> bool:
         return self.index < len(self.list)
        
    def __flattenList(self, nestedList):
        flattenedList = []

        for nestedNode in nestedList:
            if nestedNode.isInteger():
                flattenedList.append(nestedNode.getInteger())
                continue
            nestedSubList = self.__flattenList(nestedNode.getList())
            flattenedList.extend(nestedSubList)
        
        return flattenedList

# Version II
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # revesed func generates an iterator. Need to use it with a list func.
        self.stack = list(reversed(nestedList))
    
    def next(self) -> int:
        if len(self.stack) == 0:
            return
        self.__processNestedList()
        return self.stack.pop().getInteger()
        
    def __processNestedList(self):
        while self.stack and not self.stack[-1].isInteger():
            self.stack.extend(reversed(self.stack.pop().getList()))

    def hasNext(self) -> bool:
        self.__processNestedList()
        return len(self.stack) > 0

# 347 Top K Frequent Elements *
# Version I O(nlgn) O(n)
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = collections.defaultdict(int)
        pq = []
        topKFreqNums = []

        for num in nums:
            counter[num] += 1
        
        for number, freq in counter.items():
            heapq.heappush(pq, (-freq, number))
        
        while len(topKFreqNums) < k and len(pq) > 0:
            freq, number = heapq.heappop(pq)
            topKFreqNums.append(number)
        
        return topKFreqNums

# Version II Quick Select O(n) average O(n)
class Solution:
    def topKFrequent(self, nums, k):
        counter = collections.defaultdict(int)

        for num in nums:
            counter[num] += 1

        # remove duplicates since we need a distinct number
        uniques = list(counter.keys())
        self.quickSelect(0, len(uniques) - 1, uniques, counter, k)

        return uniques[:k]

    def quickSelect(self, start, end, nums, counter, k):
        if start >= end:
            return

        left, right, pivot = start, end, nums[start + (end - start) // 2]

        while left <= right:
            while left <= right and counter[nums[left]] > counter[pivot]:
                left += 1
            while left <= right and counter[pivot] > counter[nums[right]]:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        if right - start + 1 >= k:
            return self.quickSelect(start, right, nums, counter, k)
        return self.quickSelect(left, end, nums, counter, k - (left - start))


# 215 Kth Largest Element in an Array *
# Not distinct number, if it is distinct number, we need to remove duplicats
# Version I O(nlgn) O()
class Solution:
    def findKthLargest(self, nums, k):
        nums.sort(reverse=True)
        return nums[k - 1]

# Version II heap sort O(nlgk) O(K)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heap[0]

# Version III quick select O(n) - O(n**2)
class Solution:
    def findKthLargest(self, nums, k):
        self.quickSelect(0, len(nums) - 1, nums, k)
        return nums[k - 1]

    def quickSelect(self, start, end, nums, k):
        if start >= end:
            return

        left, right, pivot = start, end, nums[start + (end - start) // 2]

        while left <= right:
            while left <= right and nums[left] > pivot:
                left += 1
            while left <= right and nums[right] < pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        if right - start + 1 >= k:
            return self.quickSelect(start, right, nums, k)
        self.quickSelect(left, end, nums, k - (left - start))

# 380 Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        # dic key number value index
        # set array to generate a random value based on generated randomized index
        self.set = []
        self.dic = dict()

    def insert(self, val: int) -> bool:
        if val in self.dic:
            return False

        self.set.append(val)
        self.dic[val] = len(self.set) - 1

        return True

    def remove(self, val: int) -> bool:
        if val not in self.dic:
            return False

        index = self.dic[val]
        lastValue = self.set[-1]
        self.set[-1], self.set[index] = val, lastValue
        self.dic[lastValue] = index
        self.dic.pop(val)
        self.set.pop()
        
        return True
        
    def getRandom(self) -> int:
        randomizedIndex = random.randint(0, len(self.set) - 1)
        
        return self.set[randomizedIndex]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# 528 Random Pick with Weight
class Solution:
    def __init__(self, w: List[int]):
        self.weightArr = [0 for _ in range(len(w))]
        for index in range(len(w)):
            if index == 0:
                self.weightArr[index] = w[index]
                continue
            self.weightArr[index] = self.weightArr[index - 1] + w[index]
        

    def pickIndex(self) -> int:
        # start with 1 since 0 is not possible
        randomWeight = random.randint(1, self.weightArr[-1])
        return self.__findIndex(randomWeight)
    
    def __findIndex(self, target):
        start, end = 0, len(self.weightArr) - 1
        
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self.weightArr[mid] == target:
                return mid
            elif self.weightArr[mid] < target:
                start = mid
            else:
                end = mid
        
        if self.weightArr[start] >= target:
            return start
        return end

# 981 Time Based Key-Value Store
class TimeMap:

    def __init__(self):
        self.map = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # Assume there's no case that updates value with existing key and timestamp
        self.map[key].append((value, timestamp))

    def get(self, key: str, timestamp: int) -> str:
        arr = self.map[key]
        if len(arr) == 0:
            return ""
        
        start, end = 0, len(arr) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if arr[mid][1] == timestamp:
                return arr[mid][0]
            elif arr[mid][1] < timestamp:
                start = mid
            else:
                end = mid

        if timestamp < arr[start][1]:
            return ""
        if timestamp < arr[end][1]:
            return arr[start][0]
        return arr[end][0]
        

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

# 91 Decode Ways
# Version I DFS O(n) O(n)
class Solution:
    def numDecodings(self, s: str) -> int:
        counter = collections.defaultdict(int)
        return self.dfs(0, s, counter)

    def dfs(self, position, s, counter):
        if position == len(s):
            return 1
        # if cur char is 0, there's no combinations after
        if s[position] == "0":
            return 0
        if position in counter:
            return counter[position]

        # branch out here, but only two situations there
        count = self.dfs(position + 1, s, counter)
        if position + 1 < len(s) and int(s[position: position + 2]) <= 26:
            count += self.dfs(position + 2, s, counter)

        counter[position] = count
        return count

# Version II Dynamic programming ##

# 455 Assign Cookies
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        if len(g) == 0 or len(s) == 0:
            return 0

        sortedG = sorted(g)
        sortedS = sorted(s)
        sIndex = 0
        count = 0

        for content in sortedG:
            while sIndex < len(sortedS) and sortedS[sIndex] < content:
                sIndex += 1
            if sIndex == len(sortedS):
                break
            sIndex += 1
            count += 1
        
        return count

# 1642 Larget Substring Between Two Equal Characters
class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        indexMap = collections.defaultdict(int)
        maxLength = float('-inf')

        for i in range(len(s)):
            ch = s[i]
            maxLength = max(i - indexMap.get(ch, i) - 1, maxLength)
            if ch not in indexMap:
                indexMap[ch] = i
        
        return maxLength

# 856 Score of Parenthese
# O(n) O(n)
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        return self.dfs(0, s)[0]
    
    def dfs(self, index, s):
        count = 0
        value = 0
        curIndex = index

        while curIndex < len(s):
            ch = s[curIndex]
            if ch == "(":
                if count == 1:
                    nxtValue, nxtIndex = self.dfs(curIndex, s)
                    value += nxtValue * 2
                    # Update curIndex and count
                    curIndex = nxtIndex + 1
                    count = 0
                    continue
                count += 1
                curIndex += 1
            else:
                if count == 0:
                    return (value, curIndex)
                count -= 1
                curIndex += 1
                value += 1
        
        return (value, curIndex)

# 394 Decode String
class Solution:
    def decodeString(self, s: str) -> str:
        return self.dfs(0, s)[0]
    
    def dfs(self, position, s):
        number = 0
        string = ""
        index = position

        while index < len(s):
            ch = s[index]
            if ch.isdigit():
                number = number * 10 + int(ch)
                index += 1
            elif ch == '[':
                nxtStr, nxtPos = self.dfs(index + 1, s)
                index = nxtPos + 1
                for _ in range(number):
                    string += nxtStr
                number = 0
            elif ch == ']':
                return (string, index)
            else:
                string += ch
                index += 1
        
        return (string, index)

# 698 Partition to K equal sum subsets **
# Version I DFS O(k*N!) O(N)
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        total = 0
        for num in nums:
            total += num

        if total % k != 0:
            return False
        
        # Sort
        return self.dfs(sorted(nums, reverse=True), total // k, 0, k, set(), 0)
    
    def dfs(self, nums, target, curTotal, k, visited, position):
        if k == 1:
            return True
        
        if target == curTotal:
            return self.dfs(nums, target, 0, k - 1, visited, 0)
        
        if curTotal > target:
            return False
        
        for i in range(position, len(nums)):
            if i in visited:
                continue
            if target < nums[i]:
                return False
            visited.add(i)
            if self.dfs(nums, target, curTotal + nums[i], k, visited, i + 1):
                return True
            visited.remove(i)
        
        return False

# Version II DFS with memo
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        total = 0
        for num in nums:
            total += num

        if total % k != 0:
            return False
        
        return self.dfs(sorted(nums, reverse=True), total // k, 0, k, set(), 0, dict())
    
    def dfs(self, nums, target, curTotal, k, visited, position, memo):
        if k == 1:
            return True
        
        if curTotal > target:
            return False

        visitedStr = "".join(str(i) for i in visited) + "-" + str(k)
        if visitedStr in memo:
            return memo[visitedStr]

        if target == curTotal:
            memo[visitedStr] = self.dfs(nums, target, 0, k - 1, visited, 0, memo)
            return memo[visitedStr]
        
        for i in range(position, len(nums)):
            if i in visited:
                continue
            if target < nums[i]:
                return False
            visited.add(i)
            canPartition = self.dfs(nums, target, curTotal + nums[i], k, visited, i + 1, memo)
            if canPartition:
                memo[visitedStr] = True
                return True
            visited.remove(i)
        
        memo[visitedStr] = False
        return False

# 2610
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        counterMap = collections.defaultdict(int)
        for num in nums:
            counterMap[num] += 1
        
        matrix = []
        removed = set()
        while len(counterMap) > len(removed):
            arr = []
            for key in counterMap.keys():
                if key in removed:
                    continue
                arr.append(key)
                counterMap[key] -= 1
                if counterMap[key] == 0:
                    removed.add(key)
            matrix.append(arr)
        return matrix

# 2870 Minimum Number of Operations to Make Array Empty
# Version I 
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counter = collections.defaultdict(int)
        for num in nums:
            counter[num] += 1
        
        count = 0
        for value in counter.values():
            subCount = self.makeEmpty(value)
            if subCount == float('inf'):
                return -1
            count += subCount
        return count
    
    def makeEmpty(self, target):
        if target == 2 or target == 3:
            return 1
        if target < 0:
            return float('inf')
        
        stepWithTwoCount = self.makeEmpty(target - 2)
        stepWithThreeCount = self.makeEmpty(target - 3)
        
        return min(stepWithTwoCount, stepWithThreeCount) + 1

# Version II with memo
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counter = collections.defaultdict(int)
        for num in nums:
            counter[num] += 1
        
        count = 0
        for value in counter.values():
            subCount = self.makeEmpty(value, dict())
            if subCount == float('inf'):
                return -1
            count += subCount
        return count
    
    def makeEmpty(self, target, memo):
        if target == 2 or target == 3:
            return 1
        if target < 0:
            return float('inf')
        if target in memo:
            return memo[target]
        
        stepWithTwoCount = self.makeEmpty(target - 2, memo)
        stepWithThreeCount = self.makeEmpty(target - 3, memo)
        count = min(stepWithTwoCount, stepWithThreeCount) + 1
        memo[target] = count
        
        return count

# Version III
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counter = Counter(nums)
        ans = 0
        for c in counter.values():
            if c == 1: 
                return -1
            # floor
            ans += ceil(c / 3)
        return ans

# 2125 Number of Laser Beams in a bank
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        counter = []

        for index in range(len(bank)):
            count = 0
            for cell in bank[index]:
                if cell == '0':
                    continue
                count += 1
            if count != 0:
                counter.append(count)
        
        if len(counter) <= 1:
            return 0

        curCount = counter[0]
        nxtCount = counter[1]
        laserBeams = curCount * nxtCount

        for i in range(1, len(counter) - 1):
            curCount = counter[i]
            nxtCount = counter[i + 1]
            laserBeams += curCount * nxtCount
        
        return laserBeams

# 93 Restore IP Address
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        ips = []
        self.generateIps(0, 0, s, ips, [])
        return [".".join(ip) for ip in ips]
    
    def generateIps(self, position, curInteger, s, ips, ip):
        if curInteger == 4:
            if position == len(s):
                ips.append([integer for integer in ip])
            return

        for i in range(position, len(s)):
            integer = s[position: i + 1]
            # remove cases like 010 and 300
            if (i != position and integer[0] == '0') or int(integer) > 255:
                break
            ip.append(integer)
            self.generateIps(i + 1, curInteger + 1, s, ips, ip)
            ip.pop()

# 131 Palindrome Partitioning
# O(2**N * N) O(N)
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        partitions = []
        self.dfs(0, s, [], partitions)
        return partitions

    def dfs(self, position, s, partition, partitions):
        if position == len(s):
            partitions.append([ch for ch in partition])
            return

        for i in range(position, len(s)):
            prefix = s[position:i + 1]
            if not self.isPalindrome(prefix):
                continue
            partition.append(prefix)
            self.dfs(i + 1, s, partition, partitions)
            partition.pop()

    # O(n)
    def isPalindrome(self, s):
        return s == s[::-1]

# 300 Longest Increasing Subsequence *
# Version I O(n*2) O(n)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)

        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)

# Version II O(nlgn) O(n)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []

        for num in nums:
            index = self.binarySearch(sub, num)
            if index == len(sub):
                sub.append(num)
                continue
            sub[index] = num
        
        return len(sub)
    
    def binarySearch(self, sub, target):
        if len(sub) == 0:
            return 0
        
        start, end = 0, len(sub) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if sub[mid] == target:
                return mid
            elif sub[mid] < target:
                start = mid
            else:
                end = mid
            
        if sub[start] >= target:
            return start
        if sub[end] >= target:
            return end
        return end + 1

# 241
class Solution:
    def diffWaysToCompute(self, expression):
        ways = []

        for i in range(len(expression)):
            ch = expression[i]
            if ch in ["-", "+", "*"]:
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i+1:])
                for numberA in left:
                    for numberB in right:
                        if ch == "-":
                            ways.append(numberA - numberB)
                        elif ch == "+":
                            ways.append(numberA - numberB)
                        else:
                            ways.append(numberA * numberB)

        if len(ways) == 0:
            ways.append(int(expression))

        return ways

# 282

# 842 Split Array into Fibonacci Sequence
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        fiboList = []
        self.dfs(0, num, fiboList, [])

        return fiboList[0] if len(fiboList) != 0 else []
    
    def dfs(self, position, num, fiboList, candidate):
        if len(fiboList) != 0:
            return

        if position == len(num):
            if len(candidate) >= 3:
                fiboList.append([number for number in candidate])
            return
        
        for i in range(position, len(num)):
            # case like 01
            if i != position and num[position] == '0':
                break
            number = int(num[position:i+1])
            # Limit
            if number >= 2 ** 31:
                break
            if len(candidate) >= 2 and candidate[-2] + candidate[-1] != number:
                continue
            candidate.append(number)
            self.dfs(i + 1, num, fiboList, candidate)
            candidate.pop()

# 542 01 Matrix
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        queue = collections.deque()
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 0:
                    queue.append((i, j))
                    continue
                mat[i][j] = float('inf')
        
        distance = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                curPosition = queue.popleft()
                for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nxtPosition = (curPosition[0] + direction[0], curPosition[1] + direction[1])
                    if not self.shouldGoFurther(nxtPosition, mat, distance):
                        continue
                    mat[nxtPosition[0]][nxtPosition[1]] = distance + 1
                    queue.append(nxtPosition)
            # Increase distance
            distance += 1

        return mat
    
    def shouldGoFurther(self, position, mat, distance):
        i, j = position[0], position[1]
        if i < 0 or j < 0 or i >= len(mat) or j >= len(mat[i]):
            return False
        return mat[i][j] == float('inf')

#
