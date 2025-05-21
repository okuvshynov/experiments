#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string line;
    std::vector<std::vector<std::string>> data;
    size_t maxColumns = 0;

    // Read data from stdin line by line
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> row;
        size_t pos = 0;
        size_t nextPos;

        // Split the line by commas to get column values
        while ((nextPos = line.find(',', pos)) != std::string::npos) {
            row.push_back(line.substr(pos, nextPos - pos));
            pos = nextPos + 1;
        }
        
        // Add the last value after the last comma (or the whole line if no commas)
        row.push_back(line.substr(pos));
        
        // Keep track of the maximum number of columns
        maxColumns = std::max(maxColumns, row.size());
        
        // Add the row to our data matrix
        data.push_back(std::move(row));
    }

    // Output the number of rows and columns
    std::cout << "Rows: " << data.size() << std::endl;
    std::cout << "Columns: " << maxColumns << std::endl;

    return 0;
}