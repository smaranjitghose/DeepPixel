const navbar = document.getElementById("navbar");

const hamContainer = document.getElementById("nav-ham-container");
const hamTop = document.getElementById("ham-top");
const hamMiddle = document.getElementById("ham-middle");
const hamBottom = document.getElementById("ham-bottom");

const navLinks = document.getElementsByClassName("nav-link");

function toggleHamburger() {
	if (hamContainer.checked) {
		hamContainer.checked = false;

		navbar.classList.remove("navbar-show");

		hamContainer.style.right = "-80px";

		hamMiddle.style.visibility = "visible";
		hamMiddle.style.opacity = "1";
		hamMiddle.style.transform = "rotate(0deg)";

		hamTop.style.transform = "rotate(0deg) translateY(0px)";
		hamBottom.style.transform = "rotate(0deg) translateY(0px)";
	} else {
		hamContainer.checked = true;

		navbar.classList.add("navbar-show");

		hamContainer.style.right = "20px";

		hamMiddle.style.visibility = "hidden";
		hamMiddle.style.opacity = "0";
		hamMiddle.style.transform = "rotate(90deg)";

		hamTop.style.transform = "translateY(16px) rotate(135deg)";
		hamBottom.style.transform = "translateY(-16px) rotate(225deg)";
	}
}

hamContainer.addEventListener("click", e => {
	toggleHamburger();
});

for (const link of navLinks) {
	link.addEventListener("click", e => {
		toggleHamburger();
	});
}
