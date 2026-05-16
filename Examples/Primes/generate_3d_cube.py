from primestuff.primes import generate_prime_cube_plot_html

sz = 11

whl = sz, sz, sz

meta = generate_prime_cube_plot_html(
    output_path="Examples/Primes/primes_3d_cube.html",
    plane_width=whl[0],
    plane_height=whl[1],
    layers=whl[2],
    title=f"Prime Cube {whl[0]}x{whl[1]}x{whl[2]}",
)

print(meta)
