#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <iostream>

int main()
{
    // Create an 800x600 window with the title "SFML 2.6.2 Test"
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML 2.6.2 Test");

    // Create a circle shape with a radius of 100 pixels and set its fill color to green
    sf::CircleShape circle(100.f);
    circle.setFillColor(sf::Color::Red);
    // Position the circle roughly in the center of the window
    circle.setPosition(350.f, 250.f);



    // Main loop: runs until the window is closed
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close the window when the close event is received
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();      // Clear the screen
        window.draw(circle); // Draw the circle
        window.display();    // Display the updated frame
    }

    return 0;
}